pub mod unary {
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/unary.rs"));
}

pub mod arity1 {
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/arity1.rs"));
}

pub mod arity2 {
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/arity2.rs"));
}
