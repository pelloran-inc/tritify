//! The digit backend, on the host executor, gives the corpus hash, holds residency (also evicting on
//! every call) and the bound case; each planted defect is caught. With the `cuda` feature and a device, the same runs
//! on the device.

use super::*;
use kt_core::backend::Portable;
use kt_cuda::{HostExec, Plant, TcLinear};

const CORPUS_HASH: &str = "781285bb99bcd9cb5233f708466676db41e717b6971ce70508b57ed291478352";

fn same_as_portable(b: &Arc<dyn LinearBackend>, c: &Case) -> bool {
    b.linear_weights(&c.x, kt_core::backend::Weights { id: c.id, codes: &c.w }, c.rows, c.out, c.k)
        == Portable.linear(&c.x, &c.w, c.rows, c.out, c.k)
}

#[test]
fn the_digit_backend_gives_the_corpus_hash_and_holds_residency_and_the_bound() {
    let mut good: Vec<Arc<dyn LinearBackend>> =
        vec![Arc::new(kt_cuda::emulator()), Arc::new(kt_cuda::emulator().with_budget(1))];
    good.extend(kt_cuda::open(None, None));
    good.extend(kt_cuda::open(None, Some(1)));
    let (cases, bound) = (corpus(), flush_case());
    let mut backends: Vec<Arc<dyn LinearBackend>> = vec![Arc::new(Portable)];
    backends.extend(good.iter().cloned());
    let reports = equality_gate(&backends, &cases)
        .unwrap_or_else(|r| panic!("diverged: {:?}", r.iter().map(|g| (&g.backend, &g.diverged)).collect::<Vec<_>>()));
    for r in &reports {
        assert_eq!(r.hash, CORPUS_HASH, "{}", r.backend);
    }
    for b in &good {
        residency_gate(b, &cases).unwrap_or_else(|e| panic!("{e}"));
        assert!(same_as_portable(b, &bound), "{}: {}", b.name(), bound.label);
        println!("digit backend: {} gives {CORPUS_HASH}, holds residency and the bound case", b.name());
    }
    assert_eq!(kt_cuda::fallbacks(), 0, "no call of the corpus is outside the bounds");
}

/// One planted defect on each digit executor this host has: the host executor always, the device with the `cuda`
/// feature and a device.
fn planted(p: Plant) -> Vec<Arc<dyn LinearBackend>> {
    let mut v: Vec<Arc<dyn LinearBackend>> = vec![Arc::new(TcLinear::planted(HostExec, p))];
    v.extend(kt_cuda::open_planted(None, p));
    v
}

#[test]
fn every_planted_digit_defect_is_caught() {
    let (cases, bound) = (corpus(), flush_case());
    let gate = |b: Arc<dyn LinearBackend>| equality_gate(&[Arc::new(Portable), b], &cases);
    for b in planted(Plant::DropPlane22).into_iter().chain(planted(Plant::WrongCarry)) {
        let Err(err) = gate(b.clone()) else { panic!("{} must be caught", b.name()) };
        println!("equality gate: {} diverges on {:?}", err[1].backend, err[1].diverged);
    }
    // without the flush the corpus (k ≤ 4099) is still exact; the bound case is not
    for b in planted(Plant::NoFlush) {
        assert!(gate(b.clone()).is_ok(), "{}: the corpus never reaches the bound", b.name());
        assert!(!same_as_portable(&b, &bound), "{}: the bound case must catch a missing flush", b.name());
        println!("bound case: {} differs, the corpus does not", b.name());
    }
    for b in planted(Plant::CacheByShape) {
        let err = residency_gate(&b, &cases).expect_err("a shape-keyed cache must be caught");
        println!("residency gate: {err}");
    }
}
