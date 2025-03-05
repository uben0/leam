# LEAM

A statically typed, functional programming language inspired by the Gleam lanugage. I am basically trying to re-implement its compiler to target binaries instead of Erlang or Javascript.

```gleam
type List(a) {
  Head(a, List(a))
  Tail()
}

fn main() {
  let list = Head(4, Head(0, Tail()))
  print_list(list)
}

fn print_list(list: List(a)) {
  case list {
    Head(elem, tail) -> {
      write elem
      print_list(tail)
    }
    Tail() -> {}
  }
}
```

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

A `module` is a list of functions, identified by their position in the list. The first function is always the main function.

Values can be assigned to memory cells. A memory cell is identified by an index.

Functions assign each of their inputs to a cell. A cell can be assigned with the `assign` instruction.

A value from a memory cell can be read with `read`.

The `input` and `output` instruction will read and write a value from the standard input and from the standard output.

Values can be grouped with `group` and retreived from a group with `extract`. This is usefull for compound types.

Some logs during compilation gives good insights on what is happening:

```
STAGE EXPORTING
exporting main[] at index 0
inferring poly params
  - poly type fn('0, List('0)) -> List('0)
  - usage type fn(Int, List(Int)) -> List(Int)
exporting Head[Int32] at index 1
inferring poly params
  - poly type fn('0, List('0)) -> List('0)
  - usage type fn(Int, List(Int)) -> List(Int)
inferring poly params
  - poly type fn() -> List('0)
  - usage type fn() -> List(Int)
exporting Tail[Int32] at index 2
inferring poly params
  - poly type fn(List('0)) -> Nil
  - usage type fn(List(Int)) -> Nil
exporting print_list[Int32] at index 3
inferring poly params
  - poly type fn(List('0)) -> Nil
  - usage type fn(List('0)) -> Nil
```

# ROADMAP

- frontend
  - [ ] polymorphic let

- backend
  - [ ] alloc, ptr, free
  - [ ] block sized
