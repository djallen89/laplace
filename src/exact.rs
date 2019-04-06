use super::{L, W};
use std::f64::consts::PI as PIf64;

const FOUR_OVER_PI: f64 = 4.0 / PIf64;

pub fn exact(grid: &mut [f64], rows: usize, columns: usize, terminus: usize) {
    assert!(rows * columns == grid.len());
    assert!(terminus > 1);

    let dx = delta_e(L, columns);
    let dy = delta_e(W, rows);
    let p = terminus / 2 + terminus % 2;
    let mut ny = Vec::with_capacity(rows * p);
    let mut nx = Vec::with_capacity(columns * p);

    /* Assign elements of ny */
    for i in 0 .. rows {
        let y = e_at_i(i, dy);
        for j in 0 .. p {
            let npi = n_at_i(j) * PIf64;
            let elt = (npi * y / L).sinh() / (npi * W / L).sinh();
            ny.push(elt);
        }
    }

    /* Assign elements of nx */
    for i in 0 .. columns {
        let x = e_at_i(i, dx);
        for j in 0 .. p {
            let n = n_at_i(j);
            let elt = (n * PIf64 * x / L).sin() / n;
            nx.push(elt);
        }
    }

    sgemm(FOUR_OVER_PI, &mut ny, &mut nx, grid, rows, columns, p)
}

fn delta_e(length: f64, elts: usize) -> f64 {
    length / ((elts - 1) as f64)
}

fn n_at_i(i: usize) -> f64 {
    (i as f64) * 2.0 + 1.0
}

fn e_at_i(i: usize, de: f64) -> f64 {
    (i as f64) * de
}

fn sgemm(alpha: f64, a: &mut [f64], b: &mut [f64], c: &mut [f64],
         n: usize, m: usize, p: usize) {

    assert!(a.len() == n * p);
    assert!(b.len() == m * p);

    for i in 0 .. n {
        for j in 0 .. m {
            let mut sum = 0.0;
            for k in 0 .. p {
                sum += a[i * p + k] * b[j * p + k];
            }
            sum *= alpha;
            c[i * m + j] = sum;
        }
    }
}
