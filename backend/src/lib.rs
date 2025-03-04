use std::{io::Write, rc::Rc};

lalrpop_util::lalrpop_mod!(grammar);

pub use grammar::ModuleParser as Parser;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Type {
    Unit,
    Bool,
    Int32,
    Nat32,
    Float32,
    Fn,
    Block,
}

#[derive(Debug)]
pub struct Module {
    counter: usize,
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
    ConstUnit,
    ConstFn(usize),
    ConstBool(bool),
    ConstInt32(i32),
    ConstNat32(u32),
    ConstFloat32(f32),
    Panic,
    Push(Type, usize, Box<Self>, Box<Self>),
    Pull(Type, usize),
    Write(Type, Box<Self>),
    Read(Type),
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
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Unit,
    Bool(bool),
    Int32(i32),
    Nat32(u32),
    Float32(f32),
    Fn(usize),
    Block(Rc<[Self]>),
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
            Value::Block(_) => Type::Block,
        }
    }
    fn assert_type(self, ttype: Type) -> Self {
        assert_eq!(ttype, self.get_type());
        self
    }
}
impl Type {
    fn read(self) -> Value {
        print!("{:?} > ", self);
        std::io::stdout().flush().unwrap();
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).unwrap();
        self.parse(&buf)
    }
    fn parse(self, s: &str) -> Value {
        let s = s.trim();
        match self {
            Type::Unit => Value::Unit,
            Type::Bool => Value::Bool(s.parse().unwrap()),
            Type::Int32 => Value::Int32(s.parse().unwrap()),
            Type::Nat32 => Value::Nat32(s.parse().unwrap()),
            Type::Float32 => Value::Float32(s.parse().unwrap()),
            Type::Block => unimplemented!(),
            Type::Fn => unimplemented!(),
        }
    }
}
impl Default for Fun {
    fn default() -> Self {
        Fun {
            output_type: Type::Unit,
            inputs: Vec::new(),
            body: Inst::ConstUnit,
        }
    }
}

#[derive(Debug, Clone)]
struct Mem {
    bindings: Vec<Vec<Value>>,
}
impl Mem {
    fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }
    fn get(&self, binding: usize) -> Value {
        let Some(value) = self.bindings.get(binding).and_then(|b| b.last()) else {
            panic!("unallocated binding {:?}", binding);
        };
        value.clone()
    }
    fn push(&mut self, binding: usize, value: Value) {
        while self.bindings.get(binding).is_none() {
            self.bindings.push(Vec::new());
        }
        self.bindings[binding].push(value);
    }
    fn pop(&mut self, binding: usize) {
        self.bindings[binding].pop().unwrap();
    }
}

impl Module {
    pub const MAIN: usize = 0;

    pub fn new() -> Self {
        Self {
            counter: 0,
            functions: Vec::from([Fun::default()]),
        }
    }

    pub fn insert_fun(
        &mut self,
        index: usize,
        inputs_type: Vec<Type>,
        output_type: Type,
        body: impl Fn(ComputeWritter, &[Inst]) -> Inst,
    ) {
        let (inputs, bindings): (Vec<_>, Vec<_>) = inputs_type
            .into_iter()
            .enumerate()
            .map(|(index, ttype)| ((ttype, index), Inst::Pull(ttype, index)))
            .unzip();
        self.functions[index] = Fun {
            output_type,
            inputs,
            body: body(
                ComputeWritter {
                    counter: bindings.len(),
                },
                &bindings,
            ),
        };
    }

    pub fn reserve_fn(&mut self) -> usize {
        let index = self.functions.len();
        self.functions.push(Fun::default());
        index
    }

    pub fn set_fn(&mut self, index: usize, function: Fun) {
        *self.functions.get_mut(index).unwrap() = function;
    }

    pub fn new_id(&mut self) -> usize {
        self.counter += 1;
        self.counter
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
            mem.push(*binding, input.assert_type(*ttype));
        }
        let output = self.run_inst(&fun.body, mem);
        for (_, binding) in &fun.inputs {
            mem.pop(*binding);
        }
        output.assert_type(fun.output_type)
    }

    fn run_inst(&self, inst: &Inst, mem: &mut Mem) -> Value {
        match inst {
            Inst::ConstUnit => Value::Unit,
            Inst::ConstInt32(value) => Value::Int32(*value),
            Inst::ConstNat32(value) => Value::Nat32(*value),
            Inst::ConstFloat32(value) => Value::Float32(*value),
            Inst::ConstBool(value) => Value::Bool(*value),
            Inst::ConstFn(function) => Value::Fn(*function),
            Inst::Group(insts) => {
                Value::Block(insts.iter().map(|inst| self.run_inst(inst, mem)).collect())
            }
            Inst::Extract(t, index, value) => {
                let Value::Block(values) = self.run_inst(value, mem) else {
                    panic!("extract only valid on opaque")
                };
                let Some(value) = values.get(*index) else {
                    panic!("extract out of bound");
                };
                value.clone().assert_type(*t)
            }
            Inst::Panic => panic!(),
            Inst::Push(t, binding, head, tail) => {
                let head = self.run_inst(head, mem).assert_type(*t);
                mem.push(*binding, head);
                let tail = self.run_inst(tail, mem);
                mem.pop(*binding);
                tail
            }
            Inst::Pull(t, binding) => mem.get(*binding).assert_type(*t),
            Inst::Read(t) => t.read(),
            Inst::Write(t, inst) => {
                let value = self.run_inst(inst, mem).assert_type(*t);
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
                .assert_type(*t),
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
}

pub struct ComputeWritter {
    counter: usize,
}
impl ComputeWritter {
    pub fn push(self: Self, ttype: Type, head: Inst, tail: impl Fn(Self, Inst) -> Inst) -> Inst {
        Inst::Push(
            ttype,
            self.counter,
            Box::new(head),
            Box::new(tail(
                Self {
                    counter: self.counter + 1,
                },
                Inst::Pull(ttype, self.counter),
            )),
        )
    }
}

pub struct Writter<W: Write> {
    color: bool,
    writter: W,
    indent: usize,
}
impl<W: Write> Writter<W> {
    const NAME: &str = "\x1b[94m";
    const LABEL: &str = "\x1b[95m";
    const RESET: &str = "\x1b[0m";
    pub fn root(writter: W, root: impl Fn(Self) -> Self, color: bool) {
        let mut it = Self {
            writter,
            indent: 0,
            color,
        };
        write!(it.writter, "(").unwrap();
        it = root(it);
        writeln!(it.writter, ")").unwrap();
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
    fn name(mut self, name: &str) -> Self {
        if self.color {
            write!(self.writter, "{}{}{}", Self::NAME, name, Self::RESET).unwrap();
        } else {
            write!(self.writter, "{}", name).unwrap();
        }
        self
    }

    fn child(mut self, child: impl Writtable) -> Self {
        self.indent += 1;
        writeln!(self.writter).unwrap();
        for _ in 0..self.indent {
            write!(self.writter, "    ").unwrap();
        }
        write!(self.writter, "(").unwrap();
        self = child.write(self);
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
        w.name("module")
            .fold(&self.functions, |w, fun| w.child(fun))
    }
}
impl Writtable for Type {
    fn write<W: Write>(self, w: Writter<W>) -> Writter<W> {
        match self {
            Self::Unit => w.label("unit"),
            Self::Block => w.label("block"),
            Self::Bool => w.label("bool"),
            Self::Fn => w.label("fn"),
            Self::Int32 => w.label("i32"),
            Self::Nat32 => w.label("u32"),
            Self::Float32 => w.label("f32"),
        }
    }
}
impl Writtable for &Fun {
    fn write<W: Write>(self, writter: Writter<W>) -> Writter<W> {
        writter
            .name("fn")
            .fold(self.inputs.iter(), |w, (input_type, input_binding)| {
                w.param(*input_type).param(*input_binding)
            })
            .param(self.output_type)
            .child(&self.body)
    }
}
impl Writtable for &Box<Inst> {
    fn write<W: Write>(self, writter: Writter<W>) -> Writter<W> {
        self.as_ref().write(writter)
    }
}
impl Writtable for &Inst {
    fn write<W: Write>(self, w: Writter<W>) -> Writter<W> {
        match *self {
            Inst::Or(ref bools) => w.name("or").fold(bools, |w, cond| w.child(cond)),
            Inst::And(ref bools) => w.name("and").fold(bools, |w, cond| w.child(cond)),
            Inst::Group(ref values) => w.name("group").fold(values, |w, value| w.child(value)),
            Inst::Extract(t, index, ref block) => {
                w.name("extract").param(t).param(index).child(block)
            }
            Inst::Call(ref fun, ref params) => w
                .name("call")
                .child(fun)
                .fold(params, |w, param| w.child(param)),
            Inst::Panic => w.name("panic"),
            Inst::Eq(t, ref lhs, ref rhs) => w.name("eq").param(t).child(lhs).child(rhs),
            Inst::Branch(t, ref then, ref or) => w
                .name("branch")
                .param(t)
                .fold(then.iter(), |w, (c, v)| w.child(c).child(v))
                .child(or),
            Inst::Do(ref insts, ref tail) => w
                .name("do")
                .fold(insts, |w, inst| w.child(inst))
                .child(tail),
            Inst::Write(ttype, ref value) => w.name("write").param(ttype).child(value),
            Inst::Read(ttype) => w.name("read").param(ttype),
            Inst::Push(ttype, binding_id, ref head, ref tail) => w
                .name("push")
                .param(ttype)
                .param(binding_id)
                .child(head)
                .child(tail),
            Inst::Pull(ttype, binding_id) => w.name("pull").param(ttype).param(binding_id),
            // TODO: use a single wrapper for const
            Inst::ConstUnit => w.name("const").param(Type::Unit),
            Inst::ConstFn(function) => w.name("const").param(Type::Fn).param(function),
            Inst::ConstBool(value) => w.name("const").param(Type::Bool).param(value),
            Inst::ConstInt32(value) => w.name("const").param(Type::Int32).param(value),
            Inst::ConstNat32(value) => w.name("const").param(Type::Nat32).param(value),
            Inst::ConstFloat32(value) => w.name("const").param(Type::Float32).param(value),
            Inst::Add(ttype, ref lhs, ref rhs) => w.name("add").param(ttype).child(lhs).child(rhs),
            Inst::Sub(ttype, ref lhs, ref rhs) => w.name("sub").param(ttype).child(lhs).child(rhs),
            Inst::Mul(ttype, ref lhs, ref rhs) => w.name("mul").param(ttype).child(lhs).child(rhs),
            Inst::Div(ttype, ref lhs, ref rhs) => w.name("div").param(ttype).child(lhs).child(rhs),
        }
    }
}
