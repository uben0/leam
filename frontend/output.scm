(module
    (fn i32
        (push fn 1
            (const fn 1)
            (write i32
                (call
                    (pull fn 1)
                    (const i32 +4)))))
    (fn i32 2 i32
        (pull i32 2)))
