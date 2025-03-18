use std::{io::Write, rc::Rc};

lalrpop_util::lalrpop_mod!(grammar);

pub use grammar::ModuleParser as Parser;
use slab::Slab;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Unit,
    Bool,
    Int32,
    Nat32,
    Float32,
    Fn,
    // TODO: remove block
    Block,
    // use prod and sum instead
    Prod(Vec<Type>),
    Sum(Vec<Type>),
    Ptr,
}

#[derive(Debug)]
pub struct Module {
    functions: Vec<Fun>,
}

#[derive(Debug, PartialEq)]
pub struct Fun {
    pub output_type: Type,
    pub inputs: Vec<(Type, usize)>,
    pub body: Inst,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Inst {
    Const(Value),
    Panic,
    Assign(Type, usize, Box<Self>, Box<Self>),
    Read(Type, usize),
    Output(Type, Box<Self>),
    Input(Type),
    Do(Vec<Self>, Box<Self>),
    Eq(Type, Box<Self>, Box<Self>),
    Branch(Type, Vec<(Self, Self)>, Box<Self>),
    Call(Box<Self>, Vec<Self>),
    Group(Vec<Self>),
    Extract(Type, usize, Box<Self>),
    Add(Type, Box<Self>, Box<Self>),
    Sub(Type, Box<Self>, Box<Self>),
    Mul(Type, Box<Self>, Box<Self>),
    Div(Type, Box<Self>, Box<Self>),
    And(Vec<Self>),
    Or(Vec<Self>),
    Alloca(Box<Self>),
    Deref(Box<Self>),
    Free(Box<Self>),
}
impl Inst {
    pub fn and(mut insts: Vec<Inst>) -> Inst {
        insts.retain(|inst| !matches!(inst, Inst::Const(Value::Bool(true))));
        match insts.len() {
            0 => Inst::Const(Value::Bool(true)),
            1 => insts.pop().unwrap(),
            _ => Inst::And(insts),
        }
    }
    pub fn or(mut insts: Vec<Inst>) -> Inst {
        insts.retain(|inst| !matches!(inst, Inst::Const(Value::Bool(false))));
        match insts.len() {
            0 => Inst::Const(Value::Bool(false)),
            1 => insts.pop().unwrap(),
            _ => Inst::Or(insts),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Unit,
    Bool(bool),
    Int32(i32),
    Nat32(u32),
    Float32(f32),
    Fn(usize),
    // Block(Rc<[Self]>),
    Ptr(usize),
    Prod(Rc<[Self]>),
}
pub enum Ptr {
    Frame(usize),
    Heap(usize),
    Extract(usize, Box<Self>),
}
impl Value {
    fn get_type(&self) -> Type {
        match self {
            Value::Unit => Type::Unit,
            Value::Bool(_) => Type::Bool,
            Value::Int32(_) => Type::Int32,
            Value::Nat32(_) => Type::Nat32,
            Value::Float32(_) => Type::Float32,
            Value::Fn(_) => Type::Fn,
            // Value::Block(_) => Type::Block,
            Value::Ptr(_) => Type::Ptr,
            Value::Prod(values) => {
                // Type::Prod(values.iter().map(|value| value.get_type()).collect())
                Type::Block
            }
        }
    }
    fn assert_type(self, ttype: &Type) -> Self {
        assert_eq!(ttype, &self.get_type());
        self
    }
}
impl Type {
    fn input(&self) -> Value {
        print!("{:?} > ", self);
        std::io::stdout().flush().unwrap();
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).unwrap();
        self.parse(&buf)
    }
    fn parse(&self, s: &str) -> Value {
        let s = s.trim();
        match self {
            Type::Unit => Value::Unit,
            Type::Bool => Value::Bool(s.parse().unwrap()),
            Type::Int32 => Value::Int32(s.parse().unwrap()),
            Type::Nat32 => Value::Nat32(s.parse().unwrap()),
            Type::Float32 => Value::Float32(s.parse().unwrap()),
            Type::Block => unimplemented!(),
            Type::Fn => unimplemented!(),
            Type::Ptr => unimplemented!(),
            Type::Prod(_) => unimplemented!(),
            Type::Sum(_) => unimplemented!(),
        }
    }
}
impl Default for Fun {
    fn default() -> Self {
        Fun {
            output_type: Type::Unit,
            inputs: Vec::new(),
            body: Inst::Const(Value::Unit),
        }
    }
}

#[derive(Debug, Clone)]
struct Mem {
    bindings: Vec<Vec<Value>>,
    allocated: Slab<Value>,
}
impl Mem {
    fn new() -> Self {
        Self {
            bindings: Vec::new(),
            allocated: Slab::new(),
        }
    }
    fn get(&self, binding: usize) -> Value {
        let Some(value) = self.bindings.get(binding).and_then(|b| b.last()) else {
            panic!("unallocated binding {:?}", binding);
        };
        value.clone()
    }
    fn write(&mut self, binding: usize, value: Value) {
        // TODO: remove the push feature, replace it by function frame
        while self.bindings.get(binding).is_none() {
            self.bindings.push(Vec::new());
        }
        self.bindings[binding].push(value);
    }
    fn pop(&mut self, binding: usize) {
        self.bindings[binding].pop().unwrap();
    }

    fn allocate(&mut self, value: Value) -> Value {
        Value::Ptr(self.allocated.insert(value))
    }
    fn deref(&self, address: Value) -> Value {
        let Value::Ptr(address) = address else {
            panic!("only ptr can be derefed")
        };
        self.allocated
            .get(address)
            .expect("unallocated segment")
            .clone()
    }
    fn free(&mut self, address: Value) {
        let Value::Ptr(address) = address else {
            panic!("only ptr can be derefed")
        };
        self.allocated.remove(address);
    }
}

impl Module {
    pub const MAIN: usize = 0;

    pub fn new() -> Self {
        Self {
            functions: Vec::from([Fun::default()]),
        }
    }

    pub fn set_fn(&mut self, index: usize, function: Fun) {
        *self.functions.get_mut(index).unwrap() = function;
    }

    pub fn insert_fn(&mut self, fun: Fun) -> usize {
        let index = self.functions.len();
        self.functions.push(fun);
        index
    }

    pub fn run(&self) {
        let mut mem = Mem::new();
        let main = self.functions.get(0).unwrap();
        self.run_fun(main, Vec::new(), &mut mem);
        // TODO: check that mem is empty
    }

    fn run_fun(&self, fun: &Fun, inputs: Vec<Value>, mem: &mut Mem) -> Value {
        assert_eq!(inputs.len(), fun.inputs.len());
        for (input, (ttype, binding)) in Iterator::zip(inputs.into_iter(), fun.inputs.iter()) {
            mem.write(*binding, input.assert_type(ttype));
        }
        let output = self.run_inst(&fun.body, mem);
        for (_, binding) in &fun.inputs {
            mem.pop(*binding);
        }
        output.assert_type(&fun.output_type)
    }

    fn run_inst(&self, inst: &Inst, mem: &mut Mem) -> Value {
        match inst {
            Inst::Alloca(inst) => {
                let value = self.run_inst(inst, mem);
                mem.allocate(value)
            }
            Inst::Deref(ptr) => {
                let address = self.run_inst(ptr, mem);
                mem.deref(address)
            }
            Inst::Free(ptr) => {
                let address = self.run_inst(ptr, mem);
                mem.free(address);
                Value::Unit
            }
            Inst::Const(value) => value.clone(),
            Inst::Group(insts) => {
                Value::Prod(insts.iter().map(|inst| self.run_inst(inst, mem)).collect())
            }
            Inst::Extract(t, index, value) => {
                let Value::Prod(values) = self.run_inst(value, mem) else {
                    panic!("extract only valid on grouped values")
                };
                let Some(value) = values.get(*index) else {
                    panic!("extract out of bound");
                };
                value.clone().assert_type(t)
            }
            Inst::Panic => panic!(),
            Inst::Assign(t, binding, head, tail) => {
                let head = self.run_inst(head, mem).assert_type(t);
                mem.write(*binding, head);
                let tail = self.run_inst(tail, mem);
                mem.pop(*binding);
                tail
            }
            Inst::Read(t, binding) => mem.get(*binding).assert_type(t),
            Inst::Input(t) => t.input(),
            Inst::Output(t, inst) => {
                let value = self.run_inst(inst, mem).assert_type(t);
                println!("{:?}", value);
                value
            }
            Inst::Do(insts, tail) => {
                for inst in insts {
                    self.run_inst(inst, mem);
                }
                self.run_inst(tail, mem)
            }
            Inst::Eq(t, lhs, rhs) => {
                let lhs = self.run_inst(lhs, mem);
                let rhs = self.run_inst(rhs, mem);
                Value::Bool(match (t, lhs, rhs) {
                    (Type::Int32, Value::Int32(lhs), Value::Int32(rhs)) => lhs == rhs,
                    (Type::Nat32, Value::Nat32(lhs), Value::Nat32(rhs)) => lhs == rhs,
                    (Type::Float32, Value::Float32(lhs), Value::Float32(rhs)) => lhs == rhs,
                    (Type::Bool, Value::Bool(lhs), Value::Bool(rhs)) => lhs == rhs,
                    (Type::Unit, Value::Unit, Value::Unit) => true,
                    (t, lhs, rhs) => panic!(
                        "type error on eq of {:?} with {:?} and {:?}",
                        t,
                        lhs.get_type(),
                        rhs.get_type()
                    ),
                })
            }
            Inst::Add(t, lhs, rhs) => match (t, self.run_inst(lhs, mem), self.run_inst(rhs, mem)) {
                (Type::Int32, Value::Int32(lhs), Value::Int32(rhs)) => Value::Int32(lhs + rhs),
                (Type::Nat32, Value::Nat32(lhs), Value::Nat32(rhs)) => Value::Nat32(lhs + rhs),
                (Type::Float32, Value::Float32(lhs), Value::Float32(rhs)) => {
                    Value::Float32(lhs + rhs)
                }
                (t, lhs, rhs) => panic!("incorrect add {:?} {:?} {:?}", t, lhs, rhs),
            },
            Inst::Sub(t, lhs, rhs) => match (t, self.run_inst(lhs, mem), self.run_inst(rhs, mem)) {
                (Type::Int32, Value::Int32(lhs), Value::Int32(rhs)) => Value::Int32(lhs - rhs),
                (Type::Nat32, Value::Nat32(lhs), Value::Nat32(rhs)) => Value::Nat32(lhs - rhs),
                (Type::Float32, Value::Float32(lhs), Value::Float32(rhs)) => {
                    Value::Float32(lhs - rhs)
                }
                (t, lhs, rhs) => panic!("incorrect sub {:?} {:?} {:?}", t, lhs, rhs),
            },
            Inst::Mul(t, lhs, rhs) => match (t, self.run_inst(lhs, mem), self.run_inst(rhs, mem)) {
                (Type::Int32, Value::Int32(lhs), Value::Int32(rhs)) => Value::Int32(lhs * rhs),
                (Type::Nat32, Value::Nat32(lhs), Value::Nat32(rhs)) => Value::Nat32(lhs * rhs),
                (Type::Float32, Value::Float32(lhs), Value::Float32(rhs)) => {
                    Value::Float32(lhs * rhs)
                }
                (t, lhs, rhs) => panic!("incorrect mul {:?} {:?} {:?}", t, lhs, rhs),
            },
            Inst::Div(t, lhs, rhs) => match (t, self.run_inst(lhs, mem), self.run_inst(rhs, mem)) {
                (Type::Int32, Value::Int32(lhs), Value::Int32(rhs)) => Value::Int32(lhs / rhs),
                (Type::Nat32, Value::Nat32(lhs), Value::Nat32(rhs)) => Value::Nat32(lhs / rhs),
                (Type::Float32, Value::Float32(lhs), Value::Float32(rhs)) => {
                    Value::Float32(lhs / rhs)
                }
                (t, lhs, rhs) => panic!("incorrect div {:?} {:?} {:?}", t, lhs, rhs),
            },
            Inst::And(insts) => {
                for inst in insts {
                    let Value::Bool(value) = self.run_inst(inst, mem) else {
                        panic!("and operation only accepts booleans");
                    };
                    if !value {
                        return Value::Bool(false);
                    }
                }
                Value::Bool(true)
            }
            Inst::Or(insts) => {
                for inst in insts {
                    let Value::Bool(value) = self.run_inst(inst, mem) else {
                        panic!("or operation only accepts booleans");
                    };
                    if value {
                        return Value::Bool(true);
                    }
                }
                Value::Bool(false)
            }
            Inst::Branch(t, branches, default) => branches
                .iter()
                .find_map(|(cond, value)| {
                    let Value::Bool(cond) = self.run_inst(cond, mem) else {
                        panic!("only bool is accepted as condition");
                    };
                    cond.then(|| self.run_inst(value, mem))
                })
                .unwrap_or_else(|| self.run_inst(default, mem))
                .assert_type(t),
            Inst::Call(function, inputs) => {
                let Value::Fn(function) = self.run_inst(function, mem) else {
                    panic!("only a function can be called");
                };
                let inputs = inputs
                    .iter()
                    .map(|input| self.run_inst(input, mem))
                    .collect();
                self.run_fun(self.functions.get(function).unwrap(), inputs, mem)
            }
        }
    }

    pub fn writter(&mut self) -> Exporter {
        Exporter {
            counter: 0,
            module: self,
        }
    }
}

pub struct Exporter<'a> {
    counter: usize,
    module: &'a mut Module,
}
impl<'a> Exporter<'a> {
    pub fn write(
        &mut self,
        ttype: &Type,
        head: Inst,
        tail: impl FnOnce(Exporter, Inst) -> Inst,
    ) -> Inst {
        if let Inst::Read(p_type, _) = &head {
            assert_eq!(p_type, ttype);
            tail(
                Exporter {
                    counter: self.counter,
                    module: self.module,
                },
                head,
            )
        } else {
            Inst::Assign(
                ttype.clone(),
                self.counter,
                Box::new(head),
                Box::new(tail(
                    Exporter {
                        counter: self.counter + 1,
                        module: self.module,
                    },
                    Inst::Read(ttype.clone(), self.counter),
                )),
            )
        }
    }
    pub fn reserve_fn(&mut self) -> usize {
        let index = self.module.functions.len();
        self.module.functions.push(Fun::default());
        index
    }
    pub fn insert_fun(
        &mut self,
        index: usize,
        inputs_type: Vec<Type>,
        output_type: Type,
        mut body: impl FnMut(Exporter, Vec<Inst>) -> Inst,
    ) -> Inst {
        let (inputs, bindings): (Vec<_>, Vec<_>) = inputs_type
            .into_iter()
            .enumerate()
            .map(|(index, ttype)| ((ttype.clone(), index), Inst::Read(ttype, index)))
            .unzip();
        self.module.functions[index] = Fun {
            output_type,
            inputs,
            body: body(
                Exporter {
                    counter: bindings.len(),
                    module: self.module,
                },
                bindings,
            ),
        };
        Inst::Const(Value::Fn(index))
    }

    pub fn get(&mut self) -> Exporter {
        Exporter {
            counter: self.counter,
            module: &mut self.module,
        }
    }

    pub fn add(
        &mut self,
        ttype: Type,
        mut lhs: impl FnMut(Exporter) -> Inst,
        mut rhs: impl FnMut(Exporter) -> Inst,
    ) -> Inst {
        Inst::Add(
            ttype,
            Box::new(lhs(Exporter {
                counter: 0,
                module: self.module,
            })),
            Box::new(rhs(Exporter {
                counter: 0,
                module: self.module,
            })),
        )
    }
}

pub struct Writter<W: Write> {
    color: bool,
    writter: W,
    indent: usize,
}
// TODO: use writter for zig expressions
// pub struct ZigWritter<W: Write> {
//     color: bool,
//     writter: W,
//     indent: usize,
// }
impl<W: Write> Writter<W> {
    const NAME: &str = "\x1b[94m";
    const LABEL: &str = "\x1b[95m";
    const RESET: &str = "\x1b[0m";
    pub fn root(writter: W, root: impl Fn(Self) -> Self, color: bool) {
        root(Self {
            writter,
            indent: 0,
            color,
        });
    }
    fn param(mut self, param: impl Writtable) -> Self {
        write!(self.writter, " ").unwrap();
        param.write(self)
    }
    fn label(mut self, label: &str) -> Self {
        if self.color {
            write!(self.writter, "{}{}{}", Self::LABEL, label, Self::RESET).unwrap();
        } else {
            write!(self.writter, "{}", label).unwrap();
        }
        self
    }
    fn line(mut self) -> Self {
        writeln!(self.writter).unwrap();
        for _ in 0..self.indent {
            write!(self.writter, "    ").unwrap();
        }
        self
    }
    fn node(mut self, name: &str, inside: impl Fn(Self) -> Self) -> Self {
        self.indent += 1;
        write!(self.writter, "(").unwrap();
        if self.color {
            write!(self.writter, "{}{}{}", Self::NAME, name, Self::RESET).unwrap();
        } else {
            write!(self.writter, "{}", name).unwrap();
        }
        self = inside(self);
        write!(self.writter, ")").unwrap();
        self.indent -= 1;
        self
    }
    fn fold<T>(self, iter: impl IntoIterator<Item = T>, map: impl Fn(Self, T) -> Self) -> Self {
        iter.into_iter().fold(self, map)
    }
}

pub trait Writtable {
    fn write<W: Write>(self, writter: Writter<W>) -> Writter<W>;
}
impl Writtable for u32 {
    fn write<W: Write>(self, mut writter: Writter<W>) -> Writter<W> {
        write!(writter.writter, "{}", self).unwrap();
        writter
    }
}
impl Writtable for bool {
    fn write<W: Write>(self, mut writter: Writter<W>) -> Writter<W> {
        write!(writter.writter, "{}", self).unwrap();
        writter
    }
}
impl Writtable for usize {
    fn write<W: Write>(self, mut writter: Writter<W>) -> Writter<W> {
        write!(writter.writter, "{}", self).unwrap();
        writter
    }
}
impl Writtable for i32 {
    fn write<W: Write>(self, mut writter: Writter<W>) -> Writter<W> {
        write!(writter.writter, "{:+}", self).unwrap();
        writter
    }
}
impl Writtable for f32 {
    fn write<W: Write>(self, mut writter: Writter<W>) -> Writter<W> {
        write!(writter.writter, "{:+}", self).unwrap();
        writter
    }
}
impl Writtable for &str {
    fn write<W: Write>(self, mut writter: Writter<W>) -> Writter<W> {
        write!(writter.writter, "{}", self).unwrap();
        writter
    }
}
impl Writtable for &Module {
    fn write<W: Write>(self, w: Writter<W>) -> Writter<W> {
        w.node("module", |w| {
            w.fold(&self.functions, |w, fun| w.line().param(fun))
        })
    }
}
impl Writtable for &Type {
    fn write<W: Write>(self, w: Writter<W>) -> Writter<W> {
        match self {
            Type::Unit => w.label("unit"),
            Type::Block => w.label("block"),
            Type::Bool => w.label("bool"),
            Type::Fn => w.label("fn"),
            Type::Int32 => w.label("i32"),
            Type::Nat32 => w.label("u32"),
            Type::Float32 => w.label("f32"),
            Type::Ptr => w.label("ptr"),
            Type::Prod(ttype) => w.node("prod", |w| w.fold(ttype, |w, param| w.param(param))),
            Type::Sum(ttype) => w.node("sum", |w| w.fold(ttype, |w, param| w.param(param))),
        }
    }
}
impl Writtable for &Fun {
    fn write<W: Write>(self, writter: Writter<W>) -> Writter<W> {
        writter.node("fn", |w| {
            w.param(&self.output_type)
                .fold(self.inputs.iter(), |w, (input_type, input_binding)| {
                    w.param(input_type).param(*input_binding)
                })
                .line()
                .param(&self.body)
        })
    }
}
impl Writtable for &Box<Inst> {
    fn write<W: Write>(self, writter: Writter<W>) -> Writter<W> {
        self.as_ref().write(writter)
    }
}
impl Writtable for &Value {
    fn write<W: Write>(self, w: Writter<W>) -> Writter<W> {
        let ttype = &self.get_type();
        match self {
            Value::Unit => w.node("const", |w| w.param(ttype)),
            Value::Bool(value) => w.node("const", |w| w.param(ttype).param(*value)),
            Value::Int32(value) => w.node("const", |w| w.param(ttype).param(*value)),
            Value::Nat32(value) => w.node("const", |w| w.param(ttype).param(*value)),
            Value::Float32(value) => w.node("const", |w| w.param(ttype).param(*value)),
            Value::Fn(index) => w.node("const", |w| w.param(ttype).param(*index)),
            Value::Ptr(value) => w.node("const", |w| w.param(ttype).param(*value)),
            Value::Prod(values) => w.node("group", |w| {
                w.fold(values.iter(), |w, value| w.line().param(value))
            }),
        }
    }
}
impl Writtable for &Inst {
    fn write<W: Write>(self, w: Writter<W>) -> Writter<W> {
        // TODO: change to ref match
        match *self {
            Inst::Alloca(ref value) => w.node("alloca", |w| w.param(value)),
            Inst::Deref(ref value) => w.node("deref", |w| w.param(value)),
            Inst::Free(ref value) => w.node("free", |w| w.param(value)),
            Inst::Or(ref bools) => w.node("or", |w| w.fold(bools, |w, cond| w.line().param(cond))),
            Inst::And(ref bools) => {
                w.node("and", |w| w.fold(bools, |w, cond| w.line().param(cond)))
            }
            Inst::Group(ref values) => w.node("group", |w| {
                w.fold(values, |w, value| w.line().param(value))
            }),
            Inst::Extract(ref t, index, ref block) => {
                w.node("extract", |w| w.param(t).param(index).line().param(block))
            }
            Inst::Call(ref fun, ref params) => w.node("call", |w| {
                w.line()
                    .param(fun)
                    .fold(params, |w, param| w.line().param(param))
            }),
            Inst::Panic => w.node("panic", |w| w),
            Inst::Eq(ref t, ref lhs, ref rhs) => {
                w.node("eq", |w| w.param(t).line().param(lhs).line().param(rhs))
            }
            Inst::Branch(ref t, ref then, ref or) => w.node("branch", |w| {
                w.param(t)
                    .fold(then.iter(), |w, (c, v)| w.line().param(c).line().param(v))
                    .line()
                    .param(or)
            }),
            Inst::Do(ref insts, ref tail) => w.node("do", |w| {
                w.fold(insts, |w, inst| w.line().param(inst))
                    .line()
                    .param(tail)
            }),
            Inst::Output(ref ttype, ref value) => {
                w.node("output", |w| w.param(ttype).line().param(value))
            }
            Inst::Input(ref ttype) => w.node("input", |w| w.param(ttype)),
            Inst::Assign(ref ttype, binding_id, ref head, ref tail) => w.node("assign", |w| {
                w.param(ttype)
                    .param(binding_id)
                    .line()
                    .param(head)
                    .line()
                    .param(tail)
            }),
            Inst::Read(ref ttype, binding_id) => {
                w.node("read", |w| w.param(ttype).param(binding_id))
            }
            Inst::Const(ref value) => value.write(w),
            Inst::Add(ref ttype, ref lhs, ref rhs) => w.node("add", |w| {
                w.param(ttype).line().param(lhs).line().param(rhs)
            }),
            Inst::Sub(ref ttype, ref lhs, ref rhs) => w.node("sub", |w| {
                w.param(ttype).line().param(lhs).line().param(rhs)
            }),
            Inst::Mul(ref ttype, ref lhs, ref rhs) => w.node("mul", |w| {
                w.param(ttype).line().param(lhs).line().param(rhs)
            }),
            Inst::Div(ref ttype, ref lhs, ref rhs) => w.node("div", |w| {
                w.param(ttype).line().param(lhs).line().param(rhs)
            }),
        }
    }
}
