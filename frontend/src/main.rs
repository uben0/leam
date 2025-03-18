use backend::{Exporter, Inst, Value, Writtable};
use bevy_ecs::prelude::*;
use clap::Parser;
use std::{
    collections::{HashMap, hash_map::Entry},
    fmt::Display,
    hash::Hash,
    ops::Index,
    path::PathBuf,
    rc::Rc,
};
lalrpop_util::lalrpop_mod!(grammar);
pub(crate) use Entity;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const RED: &str = "\x1b[91m";
const GREEN: &str = "\x1b[92m";
const YELLOW: &str = "\x1b[93m";
const BLUE: &str = "\x1b[94m";
const PURPLE: &str = "\x1b[95m";

// TODO: make all component cheaply clonable with Rc

/// A text span in the input file
///
/// Instead of storing strings, it cheaply delimit with two cursor the desired text
#[derive(Component, Debug, Clone, Copy)]
pub struct Span(usize, usize);

impl Index<Span> for str {
    type Output = str;

    fn index(&self, Span(l, r): Span) -> &Self::Output {
        &self[l..r]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CompileStage {
    /// Building the syntax tree
    Inserting,
    /// Linking identifer with their declaration
    Binding,
    /// Determining type of all values
    Typing,
    /// Producing low-level representation
    Exporting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeEntity(Entity);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PatternEntity(Entity);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindingEntity(Entity);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypeEntity(Entity);

#[derive(Component, Debug)]
enum Compute {
    Nil,
    Int(i32),
    Float(f32),
    Bool(bool),
    Add(ComputeEntity, ComputeEntity),
    Sub(ComputeEntity, ComputeEntity),
    Mul(ComputeEntity, ComputeEntity),
    Div(ComputeEntity, ComputeEntity),
    Let {
        binding_ident: Span,
        binding: BindingEntity,
        head: ComputeEntity,
        tail: ComputeEntity,
    },
    Ident {
        binding_ident: Span,
        binding: Option<BindingEntity>,
    },
    Lambda {
        binding_ident: Span,
        binding: BindingEntity,
        body: ComputeEntity,
    },
    /// Prints the value to stdout
    Write(ComputeEntity),
    Chain(Vec<ComputeEntity>, ComputeEntity),
    /// Parse a value from stdin
    Read,
    Case {
        on: ComputeEntity,
        branches: Vec<(PatternEntity, ComputeEntity)>,
    },
    Call(ComputeEntity, Vec<ComputeEntity>),
}

#[derive(Component, Debug, Clone)]
enum Pattern {
    Int(i32),
    Any,
    Ident {
        binding_ident: Span,
        binding: BindingEntity,
    },
    Variant {
        label_ident: Span,
        fields_pattern: Vec<PatternEntity>,
        index: Option<usize>,
        type_def: Option<Entity>,
    },
}

#[derive(Debug, Clone)]
struct FunParam {
    binding_ident: Span,
    binding: BindingEntity,
    type_expr: Option<TypeExpr>,
}

#[derive(Component, Debug)]
struct Fun {
    binding_ident: Span,
    binding: BindingEntity,
    inputs: Vec<FunParam>,
    output_type_expr: Option<TypeExpr>,
    body: ComputeEntity,
}

#[derive(Debug, Clone)]
pub enum TypeExpr {
    Name(Span, Vec<Self>),
    Var(Span),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PolyContext {
    poly_type: TypeEntity,
    params: Vec<backend::Type>,
}
impl PolyContext {
    fn new(world: &World, poly_type: TypeEntity, params: Vec<backend::Type>) -> Self {
        assert!(world.get::<Polymorph>(poly_type.0).is_some());
        assert_eq!(
            params.len(),
            world.get::<Polymorph>(poly_type.0).unwrap().params.len()
        );
        Self { poly_type, params }
    }
}

#[derive(Component, Debug, Clone, PartialEq, Eq, Hash)]
struct Polymorph {
    params: Vec<TypeEntity>,
}

#[derive(Debug, Component, Clone, Copy, PartialEq, Eq, Hash)]
struct HasType(TypeEntity);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TypeState {
    Parameter { poly: TypeEntity, index: usize },
    Unknown,
    App(TypeTerm, Vec<TypeEntity>),
}
#[derive(Debug, Component, Clone, PartialEq, Eq, Hash)]
struct Type {
    type_of: Vec<Entity>,
    parents: Vec<(TypeEntity, usize)>,
    state: TypeState,
}

#[derive(Component, PartialEq, Eq, Debug, Clone, Copy, Hash)]
enum TypeTerm {
    Nil,
    Int,
    Bool,
    Float,
    Fn,
    Def(Entity),
}

#[derive(Debug, Clone, Copy)]
enum Ref {
    Inline(fn(&mut Module) -> Compute),
    Binding(BindingEntity),
    Variant { index: usize, type_def: Entity },
}

#[derive(Debug, Clone, Copy)]
enum BindingOrigin {
    FunctionDecl(Entity),
    Simple,
    Variant { type_def: Entity, index: usize },
}

#[derive(Component)]
struct Binding {
    origin: BindingOrigin,
    uses: Vec<ComputeEntity>,
}

#[derive(Debug, Clone)]
struct Variant {
    binding_ident: Span,
    binding: BindingEntity,
    fields_type_expr: Vec<TypeExpr>,
}

#[derive(Component)]
struct TypeDef {
    type_params: Vec<Span>,
    binding_ident: Span,
    variants: Vec<Variant>,
}

struct Separator<T> {
    need_sep: bool,
    sep: &'static str,
    iter: T,
}
trait SeparatorExt: Sized {
    fn sep(self, sep: &'static str) -> Separator<Self>;
}
impl<T> SeparatorExt for T
where
    T: Iterator,
{
    fn sep(self, sep: &'static str) -> Separator<Self> {
        Separator {
            need_sep: false,
            sep,
            iter: self,
        }
    }
}
impl<T> Iterator for Separator<T>
where
    T: Iterator,
{
    type Item = (T::Item, &'static str);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|elem| {
            if self.need_sep {
                (elem, self.sep)
            } else {
                self.need_sep = true;
                (elem, "")
            }
        })
    }
}

struct DisplayList<T> {
    group: Vec<T>,
    sep: &'static str,
}
impl<T> DisplayList<T> {
    fn new(sep: &'static str, group: impl IntoIterator<Item = T>) -> Self {
        Self {
            group: group.into_iter().collect(),
            sep,
        }
    }
}
impl<T> Display for DisplayList<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (elem, sep) in self.group.iter().sep(self.sep) {
            write!(f, "{}{}", sep, elem)?;
        }
        Ok(())
    }
}

struct DisplayType<'a> {
    module: &'a Module,
    ttype: TypeEntity,
}
impl<'a> Display for DisplayType<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.module.get::<Type>(self.ttype.0).state {
            TypeState::Parameter { index, .. } => write!(f, "{RED}'{}{RESET}", index),
            TypeState::Unknown => write!(f, "{YELLOW}{}{RESET}", self.ttype.0.index()),
            TypeState::App(term, ref params) => match (term, params.as_slice()) {
                (TypeTerm::Nil, []) => write!(f, "{BLUE}Nil{RESET}"),
                (TypeTerm::Int, []) => write!(f, "{BLUE}Int{RESET}"),
                (TypeTerm::Float, []) => write!(f, "{BLUE}Float{RESET}"),
                (TypeTerm::Bool, []) => write!(f, "{BLUE}Bool{RESET}"),
                (TypeTerm::Def(type_def), params) => {
                    write!(
                        f,
                        "{GREEN}{}{RESET}(",
                        &self.module.content[self.module.get::<TypeDef>(type_def).binding_ident]
                    )?;
                    for (param, sep) in params.iter().sep(", ") {
                        write!(
                            f,
                            "{}{}",
                            sep,
                            DisplayType {
                                module: self.module,
                                ttype: *param
                            },
                        )?;
                    }
                    write!(f, ")")
                }
                (TypeTerm::Fn, [inputs @ .., output]) => {
                    write!(f, "fn(")?;
                    for (input, sep) in inputs.iter().sep(", ") {
                        write!(
                            f,
                            "{}{}",
                            sep,
                            DisplayType {
                                module: self.module,
                                ttype: *input
                            },
                        )?;
                    }
                    write!(
                        f,
                        ") -> {}",
                        DisplayType {
                            module: self.module,
                            ttype: *output
                        }
                    )
                }
                (term, params) => write!(f, "unknown {:?} {:?}", term, params),
            },
        }
    }
}

pub(crate) struct Module {
    compilation_stage: CompileStage,
    world: World,
    content: Rc<str>,
    type_constraints: Vec<(TypeEntity, TypeEntity)>,
}

impl Module {
    fn stage_binding(&mut self) {
        println!();
        println!("{BOLD}STAGE BINDING{RESET}");
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        self.compilation_stage = CompileStage::Binding;
    }
    fn stage_typing(&mut self) {
        println!();
        println!("{BOLD}STAGE TYPING{RESET}");
        assert_eq!(self.compilation_stage, CompileStage::Binding);
        self.compilation_stage = CompileStage::Typing;
    }
    fn stage_exporting(&mut self) {
        println!();
        println!("{BOLD}STAGE EXPORTING{RESET}");
        assert_eq!(self.compilation_stage, CompileStage::Typing);
        assert!(self.type_constraints.is_empty());
        self.compilation_stage = CompileStage::Exporting;
    }
    fn type_contains(&self, hay_stack: TypeEntity, needle: TypeEntity) -> bool {
        if hay_stack == needle {
            return true;
        }
        match self.get::<Type>(hay_stack.0).state {
            TypeState::Parameter { .. } => false,
            TypeState::Unknown => false,
            TypeState::App(_, ref params) => params
                .iter()
                .any(|param| self.type_contains(*param, needle)),
        }
    }
    fn new(content: &str) -> Self {
        Self {
            compilation_stage: CompileStage::Inserting,
            world: World::new(),
            content: content.into(),
            type_constraints: Vec::new(),
        }
    }
    fn get<C: Component>(&self, entity: Entity) -> &C {
        self.world.get(entity).unwrap()
    }
    fn insert_type_def(
        &mut self,
        binding_ident: Span,
        type_params: Vec<Span>,
        variants: Vec<(Span, Vec<TypeExpr>)>,
    ) -> Entity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        let type_def = self.world.spawn_empty().id();
        let variants = variants
            .into_iter()
            .enumerate()
            .map(|(index, (binding_ident, fields_type_expr))| {
                let binding = self.insert_binding(BindingOrigin::Variant { type_def, index });
                let variant = Variant {
                    binding_ident,
                    binding,
                    fields_type_expr,
                };
                variant
            })
            .collect();
        self.world.entity_mut(type_def).insert(TypeDef {
            type_params,
            binding_ident,
            variants,
        });
        type_def
    }
    fn insert_pattern_litt_int(&mut self, litt: i32) -> PatternEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        PatternEntity(self.world.spawn(Pattern::Int(litt)).id())
    }
    fn insert_pattern_any(&mut self) -> PatternEntity {
        PatternEntity(self.world.spawn(Pattern::Any).id())
    }
    fn insert_pattern_ident(&mut self, binding_ident: Span) -> PatternEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        let entity = self.world.spawn_empty().id();
        let binding = self.insert_binding(BindingOrigin::Simple);
        self.world.entity_mut(entity).insert(Pattern::Ident {
            binding_ident,
            binding,
        });
        PatternEntity(entity)
    }
    fn insert_pattern_variant(
        &mut self,
        label_ident: Span,
        fields_pattern: Vec<PatternEntity>,
        span: Span,
    ) -> PatternEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        PatternEntity(
            self.world
                .spawn((
                    Pattern::Variant {
                        label_ident,
                        fields_pattern,
                        index: None,
                        type_def: None,
                    },
                    span,
                ))
                .id(),
        )
    }
    fn insert_type_var(&mut self) -> TypeEntity {
        assert!(matches!(
            self.compilation_stage,
            CompileStage::Binding | CompileStage::Typing
        ));
        TypeEntity(
            self.world
                .spawn(Type {
                    type_of: Vec::new(),
                    parents: Vec::new(),
                    state: TypeState::Unknown,
                })
                .id(),
        )
    }
    fn insert_type_app(
        &mut self,
        term: TypeTerm,
        params: impl IntoIterator<Item = TypeEntity>,
    ) -> TypeEntity {
        assert!(matches!(
            self.compilation_stage,
            CompileStage::Binding | CompileStage::Typing
        ));
        let ttype = TypeEntity(self.world.spawn_empty().id());
        let params: Vec<TypeEntity> = params.into_iter().collect();
        for (index, param) in params.iter().enumerate() {
            self.world
                .get_mut::<Type>(param.0)
                .unwrap()
                .parents
                .push((ttype, index));
        }
        self.world.entity_mut(ttype.0).insert(Type {
            type_of: Vec::new(),
            parents: Vec::new(),
            state: TypeState::App(term, params),
        });
        ttype
    }
    fn insert_type_app_fn(
        &mut self,
        inputs: impl IntoIterator<Item = TypeEntity>,
        output: TypeEntity,
    ) -> TypeEntity {
        self.insert_type_app(TypeTerm::Fn, inputs.into_iter().chain([output]))
    }
    // TODO: use system to check integrity of relations, backward-forward linkage
    fn polymorphic_on_rec(
        &mut self,
        type_node: TypeEntity,
        map: &HashMap<TypeEntity, TypeEntity>,
    ) -> TypeEntity {
        match self.get::<Type>(type_node.0).state {
            TypeState::Parameter { .. } => panic!("already polymorphic"),
            TypeState::Unknown => map.get(&type_node).copied().unwrap_or(type_node),
            TypeState::App(term, ref params) => {
                let params = params.clone();
                let params: Vec<_> = params
                    .into_iter()
                    .map(|param| self.polymorphic_on_rec(param, map))
                    .collect();
                self.insert_type_app(term, params)
            }
        }
    }
    // TODO: need doc
    fn polymorphic_on(
        &mut self,
        template: TypeEntity,
        vars: impl IntoIterator<Item = TypeEntity>,
    ) -> TypeEntity {
        let vars: Vec<_> = vars.into_iter().collect();
        println!(
            "polymorphing [{}]",
            DisplayList::new(", ", vars.iter().map(|param| self.display_type(*param)))
        );
        println!("  - {}", self.display_type(template));
        let mut map: HashMap<TypeEntity, TypeEntity> = HashMap::new();
        let params: Vec<TypeEntity> = vars
            .into_iter()
            .map(|var| {
                let new_var = self.insert_type_var();
                let prev = map.insert(var, new_var);
                assert_eq!(prev, None, "duplicated vars");
                new_var
            })
            .collect();
        let poly = self.polymorphic_on_rec(template, &map);
        for (index, param) in params.iter().enumerate() {
            let state = &mut self.world.get_mut::<Type>(param.0).unwrap().state;
            assert_eq!(*state, TypeState::Unknown);
            *state = TypeState::Parameter { poly, index };
        }
        println!("  - {}", self.display_type(poly));
        self.world.entity_mut(poly.0).insert(Polymorph { params });
        poly
    }
    fn set_type(&mut self, entity: Entity, ttype: TypeEntity) {
        assert_eq!(self.compilation_stage, CompileStage::Binding);
        assert!(self.world.get::<HasType>(entity).is_none());
        // self.update_type_var_usage_inside_type_value(&type_value, entity);
        self.world.entity_mut(entity).insert(HasType(ttype));
        self.world
            .get_mut::<Type>(ttype.0)
            .unwrap()
            .type_of
            .push(entity);
    }
    fn type_entry(&mut self, entity: Entity) -> TypeEntity {
        assert!(matches!(
            self.compilation_stage,
            CompileStage::Binding | CompileStage::Typing
        ));
        if let Some(HasType(ttype)) = self.world.get(entity) {
            ttype.clone()
        } else {
            let ttype = self.insert_type_var();
            self.world.entity_mut(entity).insert(HasType(ttype));
            self.world
                .get_mut::<Type>(ttype.0)
                .unwrap()
                .type_of
                .push(entity);
            ttype
        }
    }
    fn type_equiv<const N: usize>(&mut self, types: [TypeEntity; N]) {
        assert!(matches!(
            self.compilation_stage,
            CompileStage::Binding | CompileStage::Typing
        ));
        for window in types.windows(2) {
            let [a, b] = window else { unreachable!() };
            self.type_constraints.push((*a, *b));
        }
    }
    fn insert_binding_relation(&mut self, binding: BindingEntity, usage: ComputeEntity) {
        assert_eq!(self.compilation_stage, CompileStage::Binding);
        self.world
            .get_mut::<Binding>(binding.0)
            .unwrap()
            .uses
            .push(usage);
        let mut usage = self.world.get_mut::<Compute>(usage.0).unwrap();
        let Compute::Ident { binding: bound, .. } = usage.as_mut() else {
            panic!("only ident can be bound")
        };
        if bound.is_some() {
            panic!("value already bound");
        }
        *bound = Some(binding);
    }
    fn insert_litt_int(&mut self, litt: i32, span: Span) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        ComputeEntity(self.world.spawn((Compute::Int(litt), span)).id())
    }
    fn insert_litt_float(&mut self, litt: f32, span: Span) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        ComputeEntity(self.world.spawn((Compute::Float(litt), span)).id())
    }
    fn insert_litt_nil(&mut self, span: Option<Span>) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        if let Some(span) = span {
            ComputeEntity(self.world.spawn((Compute::Nil, span)).id())
        } else {
            ComputeEntity(self.world.spawn(Compute::Nil).id())
        }
    }
    fn insert_add(&mut self, lhs: ComputeEntity, rhs: ComputeEntity, span: Span) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        ComputeEntity(self.world.spawn((Compute::Add(lhs, rhs), span)).id())
    }
    fn insert_sub(&mut self, lhs: ComputeEntity, rhs: ComputeEntity, span: Span) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        ComputeEntity(self.world.spawn((Compute::Sub(lhs, rhs), span)).id())
    }
    fn insert_mul(&mut self, lhs: ComputeEntity, rhs: ComputeEntity, span: Span) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        ComputeEntity(self.world.spawn((Compute::Mul(lhs, rhs), span)).id())
    }
    fn insert_div(&mut self, lhs: ComputeEntity, rhs: ComputeEntity, span: Span) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        ComputeEntity(self.world.spawn((Compute::Div(lhs, rhs), span)).id())
    }
    fn insert_read(&mut self) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        ComputeEntity(self.world.spawn(Compute::Read).id())
    }
    fn insert_ident(&mut self, binding_ident: Span) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        ComputeEntity(
            self.world
                .spawn((
                    Compute::Ident {
                        binding_ident,
                        binding: None,
                    },
                    binding_ident,
                ))
                .id(),
        )
    }
    fn insert_binding(&mut self, origin: BindingOrigin) -> BindingEntity {
        assert!(matches!(
            self.compilation_stage,
            CompileStage::Inserting | CompileStage::Binding
        ));
        BindingEntity(
            self.world
                .spawn(Binding {
                    uses: Vec::new(),
                    origin,
                })
                .id(),
        )
    }
    fn insert_lambda(
        &mut self,
        binding_ident: Span,
        body: ComputeEntity,
        span: Span,
    ) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        let entity = self.world.spawn_empty().id();
        let binding = self.insert_binding(BindingOrigin::Simple);
        self.world.entity_mut(entity).insert((
            Compute::Lambda {
                binding_ident,
                binding,
                body,
            },
            span,
        ));
        ComputeEntity(entity)
    }
    fn insert_chain(&mut self, mut insts: Vec<ComputeEntity>) -> ComputeEntity {
        if let Some(tail) = insts.pop() {
            if insts.is_empty() {
                tail
            } else {
                assert_eq!(self.compilation_stage, CompileStage::Inserting);
                ComputeEntity(self.world.spawn(Compute::Chain(insts, tail)).id())
            }
        } else {
            self.insert_litt_nil(None)
        }
    }
    fn insert_fn(
        &mut self,
        binding_ident: Span,
        params: Vec<(Span, Option<TypeExpr>)>,
        output_type: Option<TypeExpr>,
        body: ComputeEntity,
    ) -> Entity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        let entity = self.world.spawn_empty().id();
        let binding = self.insert_binding(BindingOrigin::FunctionDecl(entity));
        let input = params
            .into_iter()
            .map(|(binding_ident, ttype)| FunParam {
                binding_ident,
                binding: self.insert_binding(BindingOrigin::Simple),
                type_expr: ttype,
            })
            .collect();
        self.world.entity_mut(entity).insert(Fun {
            binding_ident,
            binding,
            inputs: input,
            output_type_expr: output_type,
            body,
        });
        entity
    }
    fn insert_call(
        &mut self,
        fun: ComputeEntity,
        params: Vec<ComputeEntity>,
        span: Span,
    ) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        ComputeEntity(self.world.spawn((Compute::Call(fun, params), span)).id())
    }
    fn insert_let(
        &mut self,
        binding_ident: Span,
        head: ComputeEntity,
        tail: ComputeEntity,
    ) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        let entity = self.world.spawn_empty().id();
        let binding = self.insert_binding(BindingOrigin::Simple);
        self.world.entity_mut(entity).insert(Compute::Let {
            binding_ident,
            binding,
            head,
            tail,
        });
        ComputeEntity(entity)
    }

    fn insert_write(&mut self, value: ComputeEntity) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        ComputeEntity(self.world.spawn(Compute::Write(value)).id())
    }

    fn insert_case(
        &mut self,
        on: ComputeEntity,
        branches: Vec<(PatternEntity, ComputeEntity)>,
    ) -> ComputeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Inserting);
        ComputeEntity(self.world.spawn(Compute::Case { on, branches }).id())
    }

    // TODO: use typed wrapper for typed entities
    // TODO: remove variable length for fn, use a tuple instead
    fn type_term_get_number_of_params(&self, term: TypeTerm) -> (usize, Option<usize>) {
        match term {
            TypeTerm::Nil => (0, Some(0)),
            TypeTerm::Int => (0, Some(0)),
            TypeTerm::Bool => (0, Some(0)),
            TypeTerm::Float => (0, Some(0)),
            TypeTerm::Fn => (1, None),
            TypeTerm::Def(type_def) => {
                let n = self.get::<TypeDef>(type_def).type_params.len();
                (n, Some(n))
            }
        }
    }

    fn parse_type_expr(
        &mut self,
        expr: &TypeExpr,
        context: &dyn Fn(&str) -> TypeTerm,
        poly_vars: &HashMap<&str, TypeEntity>,
    ) -> TypeEntity {
        match *expr {
            TypeExpr::Name(app_term, ref app_params) => {
                let app_term = context(&self.content[app_term]);
                let (min, max) = self.type_term_get_number_of_params(app_term);
                assert!(app_params.len() >= min);
                assert!(max.map(|max| app_params.len() <= max).unwrap_or(true));
                let app_params: Vec<_> = app_params
                    .iter()
                    .map(|param| self.parse_type_expr(param, context, poly_vars))
                    .collect();
                self.insert_type_app(app_term, app_params)
            }
            TypeExpr::Var(poly_var) => *poly_vars.get(&self.content[poly_var]).unwrap(),
        }
    }
    fn parse_type_expr_discovering<'a>(
        &mut self,
        expr: &TypeExpr,
        context: &dyn Fn(&str) -> TypeTerm,
        poly_vars: &mut HashMap<&'a str, TypeEntity>,
        content: &'a str,
    ) -> TypeEntity {
        match *expr {
            TypeExpr::Name(app_term, ref app_params) => {
                let app_term = context(&self.content[app_term]);
                let (min, max) = self.type_term_get_number_of_params(app_term);
                assert!(app_params.len() >= min);
                assert!(max.map(|max| app_params.len() <= max).unwrap_or(true));
                let app_params: Vec<_> = app_params
                    .iter()
                    .map(|param| {
                        self.parse_type_expr_discovering(param, context, poly_vars, content)
                    })
                    .collect();
                self.insert_type_app(app_term, app_params)
            }
            TypeExpr::Var(poly_var) => *poly_vars
                .entry(&content[poly_var])
                .or_insert_with(|| self.insert_type_var()),
        }
    }

    fn bind_type_def(&mut self, type_def: Entity, context: &dyn Fn(&str) -> TypeTerm) {
        assert_eq!(self.compilation_stage, CompileStage::Binding);
        let content = self.content.clone();
        let poly_params = self.get::<TypeDef>(type_def).type_params.clone();

        let mut poly_vars: HashMap<&str, TypeEntity> = HashMap::new();
        let poly_params: Vec<TypeEntity> = poly_params
            .into_iter()
            .map(|param| {
                let type_var = self.insert_type_var();
                let prev = poly_vars.insert(&content[param], type_var);
                assert_eq!(
                    prev, None,
                    "type param name {:?} is redundant",
                    &content[param]
                );
                type_var
            })
            .collect();

        for variant in self
            .world
            .get::<TypeDef>(type_def)
            .unwrap()
            .variants
            .clone()
        {
            let fields_type: Vec<_> = variant
                .fields_type_expr
                .iter()
                .map(|type_expr| self.parse_type_expr(type_expr, context, &poly_vars))
                .collect();
            let custom_type = self.insert_type_app(TypeTerm::Def(type_def), poly_params.clone());
            let binding_type = self.insert_type_app_fn(fields_type, custom_type);
            let binding_type = self.polymorphic_on(binding_type, poly_params.iter().copied());
            // TODO: create trait IsTyped, and only allow those methods for entity with this trait
            self.set_type(variant.binding.0, binding_type);
        }
    }

    fn bind_fun(&mut self, fun: Entity, context: &dyn Fn(&str) -> Ref) {
        assert_eq!(self.compilation_stage, CompileStage::Binding);
        let content = self.content.clone();
        let Fun {
            inputs: ref input,
            body,
            ..
        } = *self.get(fun);
        let input = input.clone();
        self.bind_compute(body, &|b| {
            for FunParam {
                binding_ident,
                binding,
                ..
            } in &input
            {
                if b == &content[*binding_ident] {
                    return Ref::Binding(*binding);
                }
            }
            return context(b);
        });
    }

    fn bind_pattern<'a>(
        &mut self,
        pattern: PatternEntity,
        context: &dyn Fn(&str) -> Ref,
        bound: &mut HashMap<&'a str, BindingEntity>,
        content: &'a str,
    ) {
        assert_eq!(self.compilation_stage, CompileStage::Binding);
        match *self.world.get_mut::<Pattern>(pattern.0).unwrap().as_mut() {
            Pattern::Any => {}
            Pattern::Int(_) => {}
            Pattern::Ident {
                binding_ident,
                binding,
            } => {
                let prev = bound.insert(&content[binding_ident], binding);
                if prev.is_some() {
                    panic!("identifier {:?} already used", &content[binding_ident]);
                }
            }
            Pattern::Variant {
                label_ident,
                ref fields_pattern,
                ref mut index,
                ref mut type_def,
            } => {
                let Ref::Variant {
                    index: i,
                    type_def: t,
                } = context(&content[label_ident])
                else {
                    panic!("only variants can be destructured");
                };
                *index = Some(i);
                *type_def = Some(t);
                for field_pattern in fields_pattern.clone() {
                    self.bind_pattern(field_pattern, context, bound, &content);
                }
            }
        }
    }

    fn bind_compute(&mut self, root: ComputeEntity, context: &dyn Fn(&str) -> Ref) {
        assert_eq!(self.compilation_stage, CompileStage::Binding);
        let content = self.content.clone();
        match *self.world.get(root.0).unwrap() {
            Compute::Call(fun, ref params) => {
                let params = params.clone();
                self.bind_compute(fun, context);
                for param in params {
                    self.bind_compute(param, context);
                }
            }
            Compute::Case { on, ref branches } => {
                let branches = branches.clone();
                self.bind_compute(on, context);
                for (pattern, value) in branches {
                    let mut bound = HashMap::new();
                    self.bind_pattern(pattern, context, &mut bound, &content);
                    self.bind_compute(value, &|b| {
                        if let Some(binding) = bound.get(b) {
                            Ref::Binding(*binding)
                        } else {
                            context(b)
                        }
                    });
                }
            }
            Compute::Chain(ref insts, tail) => {
                for inst in insts.clone() {
                    self.bind_compute(inst, context);
                }
                self.bind_compute(tail, context);
            }
            Compute::Nil => {}
            Compute::Read => {}
            Compute::Write(value) => self.bind_compute(value, context),
            Compute::Lambda {
                binding_ident,
                body,
                binding,
            } => {
                self.bind_compute(body, &|b| {
                    if b == &content[binding_ident] {
                        Ref::Binding(binding)
                    } else {
                        context(b)
                    }
                });
            }
            Compute::Int(_) => {}
            Compute::Bool(_) => {}
            Compute::Float(_) => {}
            Compute::Add(lhs, rhs) => {
                self.bind_compute(lhs, context);
                self.bind_compute(rhs, context);
            }
            Compute::Sub(lhs, rhs) => {
                self.bind_compute(lhs, context);
                self.bind_compute(rhs, context);
            }
            Compute::Mul(lhs, rhs) => {
                self.bind_compute(lhs, context);
                self.bind_compute(rhs, context);
            }
            Compute::Div(lhs, rhs) => {
                self.bind_compute(lhs, context);
                self.bind_compute(rhs, context);
            }
            Compute::Let {
                binding_ident,
                head,
                tail,
                binding,
            } => {
                self.bind_compute(head, context);
                self.bind_compute(tail, &|b| {
                    if b == &content[binding_ident] {
                        Ref::Binding(binding)
                    } else {
                        context(b)
                    }
                });
            }
            Compute::Ident {
                binding_ident,
                binding,
            } => {
                assert_eq!(binding, None, "ident already bound");
                match context(&content[binding_ident]) {
                    Ref::Inline(construct) => {
                        let replace = construct(self);
                        *self.world.get_mut::<Compute>(root.0).unwrap() = replace
                    }
                    Ref::Binding(binding) => self.insert_binding_relation(binding, root),
                    Ref::Variant { index, type_def } => {
                        self.insert_binding_relation(
                            self.get::<TypeDef>(type_def).variants[index].binding,
                            root,
                        );
                    }
                }
            }
        }
    }

    fn typify_function_prototype(&mut self, function: Entity, context: &dyn Fn(&str) -> TypeTerm) {
        assert_eq!(self.compilation_stage, CompileStage::Binding);
        let Fun {
            binding,
            ref inputs,
            ref output_type_expr,
            ..
        } = *self.get(function);
        let input = inputs.clone();
        let output_type_expr = output_type_expr.clone();

        let content = self.content.clone();
        let mut poly_vars: HashMap<&str, TypeEntity> = HashMap::new();
        // let mut poly_params: Vec<Entity> = Vec::new();
        let mut inputs_type = Vec::new();
        for FunParam { type_expr, .. } in input {
            if let Some(type_expr) = type_expr {
                inputs_type.push(self.parse_type_expr_discovering(
                    &type_expr,
                    context,
                    &mut poly_vars,
                    &content,
                ));
            } else {
                inputs_type.push(self.insert_type_var());
            }
        }
        let output_type = if let Some(output_type_expr) = output_type_expr {
            self.parse_type_expr_discovering(&output_type_expr, context, &mut poly_vars, &content)
        } else {
            self.insert_type_var()
        };

        let binding_type = self.insert_type_app_fn(inputs_type, output_type);
        let binding_type = self.polymorphic_on(binding_type, poly_vars.values().copied());
        self.set_type(binding.0, binding_type);
    }
    fn typify_function_body(&mut self, fun: Entity) {
        assert_eq!(self.compilation_stage, CompileStage::Typing);
        let Fun {
            binding,

            inputs: ref input,
            body,
            ..
        } = *self.get(fun);
        let input = input.clone();

        let body_type = self.typify_compute(body);

        let mut fn_type_params = Vec::new();
        for FunParam { binding, .. } in input.into_iter() {
            fn_type_params.push(self.type_entry(binding.0));
        }
        fn_type_params.push(body_type);
        let fn_type = self.insert_type_app(TypeTerm::Fn, fn_type_params);
        let ttype = self.type_entry(binding.0);
        self.type_equiv([ttype, fn_type]);
    }

    fn typify_pattern(&mut self, pattern: PatternEntity) -> TypeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Typing);
        let ttype = self.type_entry(pattern.0);
        match *self.get(pattern.0) {
            Pattern::Any => {}
            Pattern::Int(_) => {
                let int_type = self.insert_type_app(TypeTerm::Int, []);
                self.type_equiv([ttype, int_type]);
            }
            Pattern::Ident { binding, .. } => {
                let binding = self.type_entry(binding.0);
                self.type_equiv([ttype, binding]);
            }
            Pattern::Variant {
                ref fields_pattern,
                index,
                type_def,
                ..
            } => {
                let fields_pattern = fields_pattern.clone();
                let HasType(binding_type) = *self.get(
                    self.get::<TypeDef>(type_def.unwrap()).variants[index.unwrap()]
                        .binding
                        .0,
                );
                let binding_type = self.instantiate_type(binding_type);
                let inputs: Vec<_> = fields_pattern
                    .into_iter()
                    .map(|field_pattern| self.typify_pattern(field_pattern))
                    .collect();
                let fn_type = self.insert_type_app_fn(inputs, ttype);
                self.type_equiv([fn_type, binding_type]);
            }
        }
        ttype
    }

    fn type_clone_map(
        &mut self,
        ttype: TypeEntity,
        map: &HashMap<TypeEntity, TypeEntity>,
    ) -> TypeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Typing);
        if let Some(mapped) = map.get(&ttype) {
            *mapped
        } else {
            match self.get::<Type>(ttype.0).state {
                TypeState::Parameter { .. } => ttype,
                TypeState::Unknown => ttype,
                TypeState::App(term, ref params) => {
                    let params = params.clone();
                    let params: Vec<_> = params
                        .into_iter()
                        .map(|param| self.type_clone_map(param, map))
                        .collect();
                    self.insert_type_app(term, params)
                }
            }
        }
    }

    fn display_type(&self, ttype: TypeEntity) -> DisplayType {
        DisplayType {
            module: self,
            ttype,
        }
    }

    fn instantiate_type(&mut self, ttype: TypeEntity) -> TypeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Typing);
        if let Some(poly) = self.world.get::<Polymorph>(ttype.0) {
            let params = poly.params.clone();
            println!(
                "instantiating [{}]",
                DisplayList::new(", ", params.iter().map(|param| self.display_type(*param)))
            );
            println!("  - {}", self.display_type(ttype));
            let params: HashMap<_, _> = params
                .into_iter()
                .zip(std::iter::repeat_with(|| self.insert_type_var()))
                .collect();
            let instantiated = self.type_clone_map(ttype, &params);
            println!("  - {}", self.display_type(instantiated));
            instantiated
        } else {
            ttype
        }
    }

    fn typify_compute(&mut self, compute: ComputeEntity) -> TypeEntity {
        assert_eq!(self.compilation_stage, CompileStage::Typing);
        let ttype = self.type_entry(compute.0);
        match *self.get(compute.0) {
            Compute::Nil => {
                let nil_type = self.insert_type_app(TypeTerm::Nil, []);
                self.type_equiv([ttype, nil_type]);
            }
            Compute::Bool(_) => {
                let bool_type = self.insert_type_app(TypeTerm::Bool, []);
                self.type_equiv([ttype, bool_type]);
            }
            Compute::Int(_) => {
                let int_type = self.insert_type_app(TypeTerm::Int, []);
                self.type_equiv([ttype, int_type]);
            }
            Compute::Float(_) => {
                let float_type = self.insert_type_app(TypeTerm::Float, []);
                self.type_equiv([ttype, float_type]);
            }
            Compute::Add(lhs, rhs) => {
                let lhs = self.typify_compute(lhs);
                let rhs = self.typify_compute(rhs);
                self.type_equiv([lhs, ttype, rhs]);
            }
            Compute::Sub(lhs, rhs) => {
                let lhs = self.typify_compute(lhs);
                let rhs = self.typify_compute(rhs);
                self.type_equiv([lhs, ttype, rhs]);
            }
            Compute::Mul(lhs, rhs) => {
                let lhs = self.typify_compute(lhs);
                let rhs = self.typify_compute(rhs);
                self.type_equiv([lhs, ttype, rhs]);
            }
            Compute::Div(lhs, rhs) => {
                let lhs = self.typify_compute(lhs);
                let rhs = self.typify_compute(rhs);
                self.type_equiv([lhs, ttype, rhs]);
            }
            Compute::Write(value) => {
                let value_type = self.typify_compute(value);
                self.type_equiv([ttype, value_type]);
            }
            Compute::Read => {}
            Compute::Call(fun, ref params) => {
                let params = params.clone();
                let fun = self.typify_compute(fun);
                let type_app_params: Vec<_> = params
                    .into_iter()
                    .map(|param| self.typify_compute(param))
                    .chain([ttype])
                    .collect();
                let fn_type = self.insert_type_app(TypeTerm::Fn, type_app_params);
                self.type_equiv([fun, fn_type]);
            }
            Compute::Ident { binding, .. } => {
                let binding_type = self.type_entry(binding.unwrap().0);
                let binding_type = self.instantiate_type(binding_type);
                self.type_equiv([ttype, binding_type]);
            }
            Compute::Chain(ref insts, tail) => {
                for inst in insts.clone() {
                    self.typify_compute(inst);
                }
                let tail = self.typify_compute(tail);
                self.type_equiv([ttype, tail]);
            }
            Compute::Lambda { binding, body, .. } => {
                let binding = self.type_entry(binding.0);
                let body = self.typify_compute(body);
                let fn_type = self.insert_type_app(TypeTerm::Fn, [binding, body]);
                self.type_equiv([ttype, fn_type]);
            }
            Compute::Let {
                binding,
                head,
                tail,
                ..
            } => {
                let binding = self.type_entry(binding.0);
                let head = self.typify_compute(head);
                let tail = self.typify_compute(tail);
                self.type_equiv([binding, head]);
                self.type_equiv([ttype, tail]);
            }
            Compute::Case { on, ref branches } => {
                let branches = branches.clone();
                let on = self.typify_compute(on);
                for (pattern, value) in branches {
                    let pattern = self.typify_pattern(pattern);
                    let value = self.typify_compute(value);
                    self.type_equiv([on, pattern]);
                    self.type_equiv([ttype, value]);
                }
            }
        }
        ttype
    }

    fn substitute_type_var(&mut self, this: TypeEntity, by: TypeEntity) {
        assert_eq!(self.compilation_stage, CompileStage::Typing);
        println!(
            "substituting {} by {}",
            self.display_type(this),
            self.display_type(by)
        );
        assert_ne!(this, by);

        let mut by_data = self.get::<Type>(by.0).clone();
        let this_data = self.get::<Type>(this.0).clone();

        for type_of in this_data.type_of {
            *self.world.get_mut(type_of).unwrap() = HasType(by);
            by_data.type_of.push(type_of);
        }
        for (parent, index) in this_data.parents {
            let mut parent_data = self.world.get_mut::<Type>(parent.0).unwrap();
            let TypeState::App(_, params) = &mut parent_data.state else {
                panic!()
            };
            params[index] = by;
            by_data.parents.push((parent, index));
        }
        *self.world.get_mut::<Type>(by.0).unwrap() = by_data;

        match this_data.state {
            TypeState::Parameter { .. } => panic!("type is not a var"),
            TypeState::Unknown => {}
            TypeState::App(_, _) => panic!("type is not a var"),
        }

        for (lhs, rhs) in &mut self.type_constraints {
            if *lhs == this {
                *lhs = by;
            }
            if *rhs == this {
                *rhs = by;
            }
        }
    }

    fn solve_constraints(&mut self) -> Result<(), ()> {
        assert_eq!(self.compilation_stage, CompileStage::Typing);
        while let Some((lhs, rhs)) = self.type_constraints.pop() {
            if lhs == rhs {
                continue;
            };
            let lhs_data = self.get::<Type>(lhs.0).clone();
            let rhs_data = self.get::<Type>(rhs.0).clone();
            match (lhs_data.state, rhs_data.state) {
                (TypeState::Parameter { .. }, TypeState::App(_, _)) => {
                    panic!(
                        "type parameter {} is constrained by {}",
                        self.display_type(lhs),
                        self.display_type(rhs)
                    );
                }
                (TypeState::App(_, _), TypeState::Parameter { .. }) => {
                    panic!(
                        "type parameter {} is constrained by {}",
                        self.display_type(lhs),
                        self.display_type(rhs)
                    );
                }
                (TypeState::Unknown, _) => {
                    if self.type_contains(rhs, lhs) {
                        panic!("infinite type");
                    }
                    self.substitute_type_var(lhs, rhs);
                }
                (_, TypeState::Unknown) => {
                    if self.type_contains(lhs, rhs) {
                        panic!("infinite type");
                    }
                    self.substitute_type_var(rhs, lhs);
                }
                (TypeState::Parameter { .. }, TypeState::Parameter { .. }) => {
                    panic!("type parameters are equivalent")
                }
                (TypeState::App(lhs_term, lhs_params), TypeState::App(rhs_term, rhs_params)) => {
                    if lhs_term != rhs_term || lhs_params.len() != rhs_params.len() {
                        println!("mismatching types");
                        println!("  - {}", self.display_type(lhs));
                        println!("  - {}", self.display_type(rhs));
                        panic!();
                    }
                    self.type_constraints.extend(Iterator::zip(
                        lhs_params.into_iter(),
                        rhs_params.into_iter(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn machine_type(&self, poly_context: &PolyContext, ttype: TypeEntity) -> backend::Type {
        assert_eq!(self.compilation_stage, CompileStage::Exporting);
        match self.get::<Type>(ttype.0).state {
            TypeState::Unknown => backend::Type::Unit,
            TypeState::Parameter { index, poly } => {
                assert_eq!(poly, poly_context.poly_type);
                poly_context.params[index].clone()
            }
            TypeState::App(term, ref params) => match (term, params.as_slice()) {
                (TypeTerm::Int, []) => backend::Type::Int32,
                (TypeTerm::Float, []) => backend::Type::Float32,
                (TypeTerm::Bool, []) => backend::Type::Bool,
                (TypeTerm::Fn, [.., _]) => backend::Type::Fn,
                (TypeTerm::Nil, []) => backend::Type::Unit,
                (TypeTerm::Def(type_def), params) => {
                    // let mut variants = Vec::new();
                    // for variant in &self.get::<TypeDef>(type_def).variants {
                    //     let HasType(fn_type) = *self.get(variant.binding);
                    //     let (inputs, _) = self.extract_fn_type(fn_type);
                    //     let params = params
                    //         .iter()
                    //         .map(|param| self.machine_type(poly_context, *param))
                    //         .collect();
                    //     let poly_context = PolyContext::new(&self.world, fn_type, params);
                    //     variants.push(backend::Type::Prod(
                    //         inputs
                    //             .iter()
                    //             .map(|input| self.machine_type(&poly_context, *input))
                    //             .collect(),
                    //     ));
                    // }
                    // backend::Type::Prod(Vec::from([
                    //     backend::Type::Nat32,
                    //     backend::Type::Sum(variants),
                    // ]))
                    backend::Type::Block
                }
                t => panic!("non exportable type {:?}", t),
            },
        }
    }

    fn export_pattern_match_to_machine(
        &self,
        pattern: PatternEntity,
        poly_context: &PolyContext,
        on: Inst,
        on_type: backend::Type,
    ) -> Inst {
        assert_eq!(self.compilation_stage, CompileStage::Exporting);
        match *self.get(pattern.0) {
            Pattern::Any => Inst::Const(Value::Bool(true)),
            Pattern::Int(value) => Inst::Eq(
                on_type,
                Box::new(on),
                Box::new(Inst::Const(Value::Int32(value))),
            ),
            Pattern::Ident { .. } => Inst::Const(Value::Bool(true)),
            Pattern::Variant {
                ref fields_pattern,
                index,
                ..
            } => {
                let mut matchers = Vec::new();
                matchers.push(Inst::Eq(
                    backend::Type::Nat32,
                    Box::new(Inst::Const(Value::Nat32(index.unwrap() as u32))),
                    Box::new(Inst::Extract(backend::Type::Nat32, 0, Box::new(on.clone()))),
                ));
                for (index, field_pattern) in fields_pattern.iter().copied().enumerate() {
                    // let field_binding = module.new_id();
                    let HasType(field_type) = *self.get(field_pattern.0);
                    let ttype = self.machine_type(poly_context, field_type);

                    matchers.push(self.export_pattern_match_to_machine(
                        field_pattern,
                        poly_context,
                        Inst::Extract(ttype.clone(), index + 1, Box::new(on.clone())),
                        ttype,
                    ));
                }

                Inst::and(matchers)
            }
        }
    }
    fn export_pattern_bind_to_machine<'a>(
        &self,
        pattern: PatternEntity,
        poly_context: &PolyContext,
        on_inst: Inst,
        mut bindings: Box<
            dyn FnMut(BindingEntity, &PolyContext, TypeEntity, Exporter) -> Inst + 'a,
        >,
    ) -> Box<dyn FnMut(BindingEntity, &PolyContext, TypeEntity, Exporter) -> Inst + 'a> {
        assert_eq!(self.compilation_stage, CompileStage::Exporting);
        match *self.get(pattern.0) {
            Pattern::Any => bindings,
            Pattern::Int(_) => bindings,
            Pattern::Ident { binding, .. } => {
                let poly_context = poly_context.clone();
                Box::new(move |b, pc, ut, m| {
                    if b == binding && *pc == poly_context {
                        on_inst.clone()
                    } else {
                        bindings(b, pc, ut, m)
                    }
                })
            }
            Pattern::Variant {
                ref fields_pattern, ..
            } => fields_pattern.iter().copied().enumerate().fold(
                bindings,
                |bindings, (index, field_pattern)| {
                    let HasType(field_type) = *self.get(field_pattern.0);
                    let ttype = self.machine_type(poly_context, field_type);
                    self.export_pattern_bind_to_machine(
                        field_pattern,
                        poly_context,
                        Inst::Extract(ttype, index + 1, Box::new(on_inst.clone())),
                        // module,
                        bindings,
                    )
                },
            ),
        }
    }

    fn extract_fn_type(&self, fn_type: TypeEntity) -> (Vec<TypeEntity>, TypeEntity) {
        assert_eq!(self.compilation_stage, CompileStage::Exporting);
        let TypeState::App(TypeTerm::Fn, ref params) = self.get::<Type>(fn_type.0).state else {
            panic!()
        };
        let [inputs @ .., output] = params.as_slice() else {
            panic!()
        };
        (Vec::from(inputs), *output)
    }

    fn export_mono_function_to_machine(
        &self,
        function_index: usize,
        function: Entity,
        module: Exporter,
        bindings: &mut dyn FnMut(BindingEntity, &PolyContext, TypeEntity, Exporter) -> Inst,
    ) {
        let binding = self.get::<Fun>(function).binding;
        let HasType(binding_type) = *self.get(binding.0);
        // TODO: maybe use wrapper for poly types
        assert!(self.get::<Polymorph>(binding_type.0).params.is_empty());
        let poly_context = PolyContext::new(&self.world, binding_type, Vec::new());
        self.export_function_to_machine(&poly_context, function_index, function, module, bindings);
    }

    fn export_variant_constructor_to_machine(
        &self,
        poly_context: &PolyContext,
        function_index: usize,
        type_def: Entity,
        index: usize,
        mut module: Exporter,
    ) -> Inst {
        let variant = &self.get::<TypeDef>(type_def).variants[index];
        println!(
            "exporting {PURPLE}{}{:?}{RESET} at index {YELLOW}{}{RESET}",
            &self.content[variant.binding_ident], poly_context.params, function_index,
        );
        // TODO: remove ambiguity from where the poly type comes from
        let HasType(binding_type) = *self.get(variant.binding.0);
        assert_eq!(binding_type, poly_context.poly_type);
        let (inputs, output) = self.extract_fn_type(binding_type);
        let inputs: Vec<_> = inputs
            .into_iter()
            .map(|input_type| self.machine_type(&poly_context, input_type))
            .collect();
        let output_type = self.machine_type(&poly_context, output);
        module.insert_fun(function_index, inputs, output_type, |_, i| {
            Inst::Group(
                [Inst::Const(Value::Nat32(index as u32))]
                    .into_iter()
                    .chain(i)
                    .collect(),
            )
        })
    }

    // TODO: use an parametric type for compile stages, and impl only on appropritate types

    fn resolve_top_level_binding(
        &self,
        poly_context: &PolyContext,
        binding: BindingEntity,
        usage_type: TypeEntity,
        mut module: Exporter,
        bindings: &mut HashMap<(BindingEntity, PolyContext), Inst>,
    ) -> Inst {
        assert_eq!(self.compilation_stage, CompileStage::Exporting);
        let HasType(binding_type) = *self.get(binding.0);
        let context = if let Some(poly) = self.world.get::<Polymorph>(binding_type.0) {
            let mut discovered = HashMap::new();
            println!("inferring poly params");
            println!("  - poly type {}", self.display_type(binding_type));
            println!("  - usage type {}", self.display_type(usage_type));
            self.infer_poly_params_from_usage(
                binding_type,
                binding_type,
                usage_type,
                &mut discovered,
            );
            &PolyContext::new(
                &self.world,
                binding_type,
                poly.params
                    .iter()
                    .enumerate()
                    .map(|(index, _)| {
                        self.machine_type(
                            poly_context,
                            *discovered
                                .get(&index)
                                .expect("poly type parameter could not be inferred from usage"),
                        )
                    })
                    .collect(),
            )
        } else {
            poly_context
        };
        if let Some(binding) = bindings.get(&(binding, context.clone())) {
            return binding.clone();
        }
        match self.get::<Binding>(binding.0).origin {
            BindingOrigin::FunctionDecl(function) => {
                let function_index = module.reserve_fn();
                let inst = Inst::Const(Value::Fn(function_index));
                bindings.insert((binding, context.clone()), inst.clone());
                self.export_function_to_machine(
                    &context,
                    function_index,
                    function,
                    module,
                    &mut |b, pc, ut, m| self.resolve_top_level_binding(pc, b, ut, m, bindings),
                );
                inst
            }
            BindingOrigin::Variant { type_def, index } => {
                let function_index = module.reserve_fn();
                let inst = Inst::Const(Value::Fn(function_index));
                bindings.insert((binding, context.clone()), inst.clone());
                self.export_variant_constructor_to_machine(
                    &context,
                    function_index,
                    type_def,
                    index,
                    module,
                );
                inst
            }
            BindingOrigin::Simple => {
                panic!()
                // let machine_type = self.machine_type(&context, usage_type);
                // Inst::Pull(machine_type, module.new_id())
            }
        }
    }

    fn export_compute_to_machine(
        &self,
        compute: ComputeEntity,
        poly_context: &PolyContext,
        mut exporter: Exporter,
        bindings: &mut dyn FnMut(BindingEntity, &PolyContext, TypeEntity, Exporter) -> Inst,
    ) -> Inst {
        assert_eq!(self.compilation_stage, CompileStage::Exporting);
        let HasType(ttype) = *self.get(compute.0);
        match *self.get(compute.0) {
            Compute::Nil => Inst::Const(Value::Unit),
            Compute::Int(value) => Inst::Const(Value::Int32(value)),
            Compute::Float(value) => Inst::Const(Value::Float32(value)),
            Compute::Bool(value) => Inst::Const(Value::Bool(value)),
            Compute::Add(lhs, rhs) => {
                let ttype = self.machine_type(poly_context, ttype);
                Inst::Add(
                    ttype,
                    Box::new(self.export_compute_to_machine(
                        lhs,
                        poly_context,
                        exporter.get(),
                        bindings,
                    )),
                    Box::new(self.export_compute_to_machine(
                        rhs,
                        poly_context,
                        exporter.get(),
                        bindings,
                    )),
                )
            }
            Compute::Sub(lhs, rhs) => {
                let ttype = self.machine_type(poly_context, ttype);
                Inst::Sub(
                    ttype,
                    Box::new(self.export_compute_to_machine(
                        lhs,
                        poly_context,
                        exporter.get(),
                        bindings,
                    )),
                    Box::new(self.export_compute_to_machine(
                        rhs,
                        poly_context,
                        exporter.get(),
                        bindings,
                    )),
                )
            }
            Compute::Mul(lhs, rhs) => {
                let ttype = self.machine_type(poly_context, ttype);
                Inst::Mul(
                    ttype,
                    Box::new(self.export_compute_to_machine(
                        lhs,
                        poly_context,
                        exporter.get(),
                        bindings,
                    )),
                    Box::new(self.export_compute_to_machine(
                        rhs,
                        poly_context,
                        exporter.get(),
                        bindings,
                    )),
                )
            }
            Compute::Div(lhs, rhs) => {
                let ttype = self.machine_type(poly_context, ttype);
                Inst::Div(
                    ttype,
                    Box::new(self.export_compute_to_machine(
                        lhs,
                        poly_context,
                        exporter.get(),
                        bindings,
                    )),
                    Box::new(self.export_compute_to_machine(
                        rhs,
                        poly_context,
                        exporter.get(),
                        bindings,
                    )),
                )
            }
            Compute::Read => {
                let ttype = self.machine_type(poly_context, ttype);
                Inst::Input(ttype)
            }
            Compute::Write(value) => {
                let ttype = self.machine_type(poly_context, ttype);
                Inst::Output(
                    ttype,
                    Box::new(self.export_compute_to_machine(
                        value,
                        poly_context,
                        exporter,
                        bindings,
                    )),
                )
            }
            Compute::Chain(ref insts, tail) => Inst::Do(
                insts
                    .iter()
                    .map(|inst| {
                        self.export_compute_to_machine(
                            *inst,
                            poly_context,
                            exporter.get(),
                            bindings,
                        )
                    })
                    .collect(),
                Box::new(self.export_compute_to_machine(tail, poly_context, exporter, bindings)),
            ),
            Compute::Call(function, ref parameters) => Inst::Call(
                Box::new(self.export_compute_to_machine(
                    function,
                    poly_context,
                    exporter.get(),
                    bindings,
                )),
                parameters
                    .iter()
                    .map(|param| {
                        self.export_compute_to_machine(
                            *param,
                            poly_context,
                            exporter.get(),
                            bindings,
                        )
                    })
                    .collect(),
            ),
            Compute::Ident { binding, .. } => {
                bindings(binding.unwrap(), poly_context, ttype, exporter.get())
            }
            Compute::Let {
                binding,
                head,
                tail,
                ..
            } => {
                let HasType(binding_type) = *self.get(binding.0);
                let binding_type = self.machine_type(poly_context, binding_type);
                let head =
                    self.export_compute_to_machine(head, poly_context, exporter.get(), bindings);
                exporter.write(&binding_type, head, |exporter, head| {
                    self.export_compute_to_machine(
                        tail,
                        &poly_context,
                        exporter,
                        &mut |b, pc, ut, exporter| {
                            if b == binding && pc == poly_context {
                                head.clone()
                            } else {
                                bindings(b, pc, ut, exporter)
                            }
                        },
                    )
                })
            }
            Compute::Lambda { binding, body, .. } => {
                let HasType(binding_type) = *self.get(binding.0);
                let HasType(body_type) = *self.get(body.0);
                let input_type = self.machine_type(poly_context, binding_type);
                let output_type = self.machine_type(poly_context, body_type);
                let index = exporter.reserve_fn();
                exporter.insert_fun(
                    index,
                    Vec::from([input_type]),
                    output_type,
                    |exporter, mut inputs| {
                        let input = inputs.pop().unwrap();
                        self.export_compute_to_machine(
                            body,
                            poly_context,
                            exporter,
                            &mut |b, pc, ut, exporter| {
                                if b == binding && pc == poly_context {
                                    input.clone()
                                } else {
                                    bindings(b, pc, ut, exporter)
                                }
                            },
                        )
                    },
                )
            }
            Compute::Case { on, ref branches } => {
                let HasType(on_type) = *self.get(on.0);
                let branches = branches.clone();
                let on_type = self.machine_type(poly_context, on_type);
                let ttype = self.machine_type(poly_context, ttype);
                let head =
                    self.export_compute_to_machine(on, poly_context, exporter.get(), bindings);
                exporter.write(&on_type, head, |mut w, on| {
                    let mut thens: Vec<(Inst, Inst)> = Vec::new();
                    for (pattern, value) in branches.clone() {
                        let mut export_pattern_bind_to_machine = self
                            .export_pattern_bind_to_machine(
                                pattern,
                                poly_context,
                                on.clone(),
                                // w,
                                Box::new(|a, b, c, d| bindings(a, b, c, d)),
                            );
                        thens.push((
                            self.export_pattern_match_to_machine(
                                pattern,
                                poly_context,
                                on.clone(),
                                on_type.clone(),
                            ),
                            self.export_compute_to_machine(
                                value,
                                poly_context,
                                w.get(),
                                &mut export_pattern_bind_to_machine,
                            ),
                        ));
                    }
                    Inst::Branch(ttype, thens, Box::new(Inst::Panic))
                })
            }
        }
    }

    fn type_eq(&self, lhs: TypeEntity, rhs: TypeEntity) -> bool {
        match (
            &self.get::<Type>(lhs.0).state,
            &self.get::<Type>(rhs.0).state,
        ) {
            (TypeState::Unknown, TypeState::Unknown) => true,
            (lhs @ TypeState::Parameter { .. }, rhs @ TypeState::Parameter { .. }) => lhs == rhs,
            (TypeState::App(lhs_term, lhs_params), TypeState::App(rhs_term, rhs_params)) => {
                lhs_term == rhs_term
                    && Iterator::zip(lhs_params.iter(), rhs_params.iter())
                        .all(|(lhs, rhs)| self.type_eq(*lhs, *rhs))
            }
            _ => false,
        }
    }

    fn infer_poly_params_from_usage(
        &self,
        poly_data: TypeEntity,
        poly_type: TypeEntity,
        usage: TypeEntity,
        discovered: &mut HashMap<usize, TypeEntity>,
    ) {
        assert_eq!(self.compilation_stage, CompileStage::Exporting);
        match self.get::<Type>(poly_type.0).state {
            TypeState::Parameter { poly, index } => {
                assert_eq!(poly, poly_data);
                assert_eq!(poly_type, self.get::<Polymorph>(poly_data.0).params[index]);
                match discovered.entry(index) {
                    Entry::Occupied(entry) => {
                        assert!(self.type_eq(*entry.get(), usage));
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(usage);
                    }
                }
            }
            TypeState::Unknown => {}
            TypeState::App(_, ref poly_params) => {
                let TypeState::App(_, ref usage_params) = self.get::<Type>(usage.0).state else {
                    panic!()
                };
                for (poly_param, usage_param) in
                    Iterator::zip(poly_params.iter(), usage_params.iter())
                {
                    self.infer_poly_params_from_usage(
                        poly_data,
                        *poly_param,
                        *usage_param,
                        discovered,
                    );
                }
            }
        }
    }

    fn export_function_to_machine(
        &self,
        poly_context: &PolyContext,
        function_index: usize,
        function: Entity,
        mut exporter: Exporter,
        bindings: &mut dyn FnMut(BindingEntity, &PolyContext, TypeEntity, Exporter) -> Inst,
    ) -> Inst {
        assert_eq!(self.compilation_stage, CompileStage::Exporting);
        let function_data: &Fun = self.get(function);
        println!(
            "exporting {PURPLE}{}{:?}{RESET} at index {YELLOW}{}{RESET}",
            &self.content[function_data.binding_ident], poly_context.params, function_index,
        );

        let HasType(function_type) = *self.get(function_data.binding.0);
        let poly_data = self.get::<Polymorph>(function_type.0);
        assert_eq!(poly_data.params.len(), poly_context.params.len());
        assert_eq!(function_type, poly_context.poly_type);

        let (inputs_type, inputs_binding): (Vec<_>, Vec<_>) = function_data
            .inputs
            .iter()
            .map(|FunParam { binding, .. }| {
                let HasType(binding_type) = *self.get(binding.0);
                let binding_type = self.machine_type(&poly_context, binding_type);
                (binding_type, *binding)
            })
            .unzip();
        let HasType(body_type) = *self.get(function_data.body.0);
        let output_type = self.machine_type(poly_context, body_type);
        exporter.insert_fun(
            function_index,
            inputs_type,
            output_type,
            |exporter, inputs| {
                let inputs: Vec<_> =
                    Iterator::zip(inputs.iter().cloned(), inputs_binding.iter().copied()).collect();
                self.export_compute_to_machine(
                    function_data.body,
                    poly_context,
                    exporter,
                    &mut |b, pc, ut, m| {
                        for (inst, binding) in inputs.iter() {
                            if b == *binding && pc == poly_context {
                                return inst.clone();
                            }
                        }
                        bindings(b, pc, ut, m)
                    },
                )
            },
        )
    }
}

#[derive(Parser)]
struct Args {
    input: PathBuf,
}

fn main() {
    let Args { input } = Args::parse();
    let content = std::fs::read_to_string(input).unwrap();
    let content = content.as_str();
    let mut module = Module::new(content);
    let parser = grammar::ModuleParser::new();
    let top_levels = parser.parse(&mut module, &content).unwrap();
    let mut type_defs: Vec<Entity> = Vec::new();
    let mut funs: Vec<Entity> = Vec::new();
    let mut top_level_refs: HashMap<&str, Ref> = HashMap::from([
        ("Nil", Ref::Inline(|_| Compute::Nil)),
        ("True", Ref::Inline(|_| Compute::Bool(true))),
        ("False", Ref::Inline(|_| Compute::Bool(false))),
    ]);
    let mut top_level_types: HashMap<&str, TypeTerm> = HashMap::from([
        ("Nil", TypeTerm::Nil),
        ("Int", TypeTerm::Int),
        ("Bool", TypeTerm::Bool),
        ("Float", TypeTerm::Float),
    ]);
    let mut main = None;
    // TODO: despawn any unreferenced entity
    for top_level in &top_levels {
        if let Some(fun) = module.world.get::<Fun>(*top_level) {
            funs.push(*top_level);
            let ident = &content[fun.binding_ident];
            let prev = top_level_refs.insert(ident, Ref::Binding(fun.binding));
            if prev.is_some() {
                panic!("identifier {:?} used more than once", ident);
            }
            if ident == "main" {
                main = Some(*top_level);
            }
        } else if let Some(type_def) = module.world.get::<TypeDef>(*top_level) {
            type_defs.push(*top_level);
            for (index, variant) in type_def.variants.iter().enumerate() {
                let ident = &content[variant.binding_ident];
                let prev = top_level_refs.insert(
                    ident,
                    Ref::Variant {
                        index,
                        type_def: *top_level,
                    },
                );
                if prev.is_some() {
                    panic!("identifier {:?} used more than once", ident);
                }
            }
            let ident = &content[type_def.binding_ident];
            let prev = top_level_types.insert(ident, TypeTerm::Def(*top_level));
            if prev.is_some() {
                panic!("identifier {:?} used more than once", ident);
            }
        } else {
            panic!();
        }
    }

    module.stage_binding();

    for type_def in &type_defs {
        module.bind_type_def(*type_def, &|type_ident| {
            if let Some(ttype) = top_level_types.get(type_ident) {
                ttype.clone()
            } else {
                panic!("type {:?} not found in context", type_ident)
            }
        });
    }
    for fun in &funs {
        module.bind_fun(*fun, &|b| {
            if let Some(binding) = top_level_refs.get(b) {
                *binding
            } else {
                panic!("binding {} not found in context", b)
            }
        });
    }

    for fun in &funs {
        module.typify_function_prototype(*fun, &|type_ident| {
            if let Some(ttype) = top_level_types.get(type_ident) {
                ttype.clone()
            } else {
                panic!("type {:?} not found in context", type_ident)
            }
        });
    }

    module.stage_typing();

    for fun in &funs {
        module.typify_function_body(*fun);
    }
    match module.solve_constraints() {
        Ok(()) => {}
        Err(()) => panic!(),
    }

    module.stage_exporting();

    let mut machine = backend::Module::new();
    let mut bindings_map = HashMap::new();
    let mut bindings = |b: BindingEntity, pc: &PolyContext, ut, m: Exporter| {
        module.resolve_top_level_binding(pc, b, ut, m, &mut bindings_map)
    };
    // let mut types_map = HashMap::new();
    // let mut types = |
    module.export_mono_function_to_machine(
        backend::Module::MAIN,
        main.unwrap(),
        machine.writter(),
        &mut bindings,
    );

    println!();
    backend::Writter::root(&mut std::io::stdout(), |w| machine.write(w), true);
    println!();

    backend::Writter::root(
        &mut std::fs::File::create("output.scm").unwrap(),
        |w| machine.write(w),
        false,
    );

    machine.run();
}
