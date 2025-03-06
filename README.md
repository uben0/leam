# LEAM

A statically typed, functional programming language inspired by the Gleam lanugage. I am basically trying to re-implement its compiler to target binaries instead of Erlang or Javascript.

```gleam
type List(a) {
  Node(a, List(a))
  Leaf()
}

fn main() {
  let list = Node(4, Node(0, Leaf()))
  print_list(list)
}

fn print_list(list: List(a)) {
  case list {
    Node(elem, tail) -> {
      write elem
      print_list(tail)
    }
    Leaf() -> {}
  }
}
```

Some logs during compilation gives good insights on what is happening:

```
exporting main[] at index 0
inferring poly params
  - poly type fn('0, List('0)) -> List('0)
  - usage type fn(Int, List(Int)) -> List(Int)
exporting Node[Int32] at index 1
inferring poly params
  - poly type fn('0, List('0)) -> List('0)
  - usage type fn(Int, List(Int)) -> List(Int)
inferring poly params
  - poly type fn() -> List('0)
  - usage type fn() -> List(Int)
exporting Leaf[Int32] at index 2
inferring poly params
  - poly type fn(List('0)) -> Nil
  - usage type fn(List(Int)) -> Nil
exporting print_list[Int32] at index 3
inferring poly params
  - poly type fn(List('0)) -> Nil
  - usage type fn(List('0)) -> Nil
```

## Backend

The backend is a low level representation of the program.

A module contains functions. A function is identified by its index (position in the list).

The following data types are available:
- [x] `i32`
- [x] `u32`
- [x] `f32`
- [x] `unit` the void value
- [x] `fn` a function index
- [x] `block` an opaque value
    - [ ] will have a size
    - [ ] and an alignment

The following instructions are available:
- [x] `const` provides a hard written value
- [x] `assign` and `read` assigns then reads a value to and from a memory cell
    - [x] functions take an arbitrary list of parameters assigned to memory cells
    - [ ] a `write` instruction mutates the stored value (will add a flag to `assign`, either `static` or `volatile` to disallow or allow mutation)
- [x] `call` invokes a function with the provided parameters
- [x] `input` and `output` reads and writes value to stdin and stdout
- [x] `add`, `sub`, `mul`, `div` and `eq` are the usual binary operators
- [x] `and` and `or` are the boolean operators with arbitrary number of operands, currently they are lazy and will shortcircuit
    - [ ] add a flag to control wether to be lazy or greedy
- [x] `branch` is currently the only control flow available
    - execute the first branch with a positive predicate
    - has a last default branch in case non of the above were positive
- [x] `panic` does what it says

The above code compiles to:

```scheme
(module
    (fn unit
        (assign block 0
            (call
                (const fn 1)
                (const i32 +4)
                (call
                    (const fn 1)
                    (const i32 +0)
                    (call
                        (const fn 2))))
            (call
                (const fn 3)
                (read block 0))))
    (fn block i32 0 block 1
        (group
            (const u32 0)
            (read i32 0)
            (read block 1)))
    (fn block
        (group
            (const u32 1)))
    (fn unit block 0
        (branch unit
            (eq u32
                (const u32 0)
                (extract u32 0
                    (read block 0)))
            (do
                (output i32
                    (extract i32 1
                        (read block 0)))
                (call
                    (const fn 3)
                    (extract block 2
                        (read block 0))))
            (eq u32
                (const u32 1)
                (extract u32 0
                    (read block 0)))
            (const unit)
            (panic))))
```

# ROADMAP

- frontend
  - [ ] polymorphic let
  - [ ] embeded polymorphism
  - [ ] lifetime and reference counting
  - [ ] determine size of types
  - [ ] extend builtin types
    - [ ] list
    - [ ] string

- backend
  - [ ] alloc, ptr, free
  - [ ] block sized
  - [ ] emit LLVM IR
