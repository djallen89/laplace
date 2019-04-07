use std::mem;
use super::{get_idx, residual};
use super::{RES_MAX, OMEGA};

pub fn slor(matrix: &mut Vec<f64>, rows: usize, columns: usize,
            dx: f64, dy: f64) -> Vec<f64> {
    
    let mut max_residuals = Vec::with_capacity(100); // 100 is a guess for n.
    let dx_sq = dx * dx;
    let dy_sq = dy * dy;
    let inv_dx_sq = 1.0 / dx_sq;
    let inv_dy_sq = 1.0 / dy_sq;

    /* The Thomas algorithm has a forward sweep and a reverse
     * sweep. There is a data dependence in both of these steps on the
     * previous iteration in the algorithm, and a data dependence for
     * one of the columns in the forward sweep on the column. However,
     * in the first step, there is a calculation which is the same
     * regardless of which column is being calculated; this
     * calculation can be done once for the entirety of the
     * matrix. Additionally, p, q and r will also be the same for
     * every iteration. In fact, not only will they be the same for
     * every iteration, but their first elements are equal to their
     * last, and all of their middle elements are the same.
     */

    let q1 = 1.0;
    let r1 = 0.0;

    let pj = inv_dy_sq;
    let qj = -2.0 * (inv_dx_sq + inv_dy_sq);
    let rj = inv_dy_sq;

    let mut r_prime = Vec::with_capacity(rows);
    r_prime.push(r1 / q1);
    for j in 1 .. rows - 1 {
        let denominator = qj - pj * r_prime[j - 1];
        r_prime.push(rj / denominator);
    }

    /* Because the algorithm works with data in a columnar manner and
     * the data is stored in row major format, the data should be
     * transposed to facilitate its use in an efficient manner.
     */
    
    let mut matrix_n = Vec::with_capacity(rows * columns);
    let mut matrix_np1 = Vec::with_capacity(rows * columns);

    let stride_orig = columns;
    let stride_trans = rows; 
    
    /* Need to transpose the first two columns before starting the
     * algorithm on the first iteration. */

    for column in 0 .. columns {
        for row in 0 .. rows {
            let element = matrix[get_idx(row, column, stride_orig)];
            matrix_n.push(element);
            matrix_np1.push(element);
        }
    }

    let bc1 = matrix[get_idx(0, columns / 2, stride_orig)];
    let bc2 = matrix[get_idx(rows - 1, columns / 2, stride_orig)];

    let mut mat_n = &mut matrix_n;
    let mut mat_np1 = &mut matrix_np1;

    let mut even_swap_count = true;
    let mut s_prime: Vec<f64> = (0 .. rows).map(|_| 0.0 as f64).collect();
    s_prime[0] = bc1;
    s_prime[rows - 1] = bc2;

    loop {
        let mut res_max = 0.0f64;

        for i in 1 .. columns - 1 {
            /* Calculate s_prime and calculate maximum residual */
            for j in 1 .. rows - 1 {
                let u_ij = mat_n[get_idx(i, j, stride_trans)]; 

                let u_im1j_n = mat_n[get_idx(i - 1, j, stride_trans)];
                let u_im1j_np1 = mat_np1[get_idx(i - 1, j, stride_trans)];
                let u_ip1j = mat_n[get_idx(i + 1, j, stride_trans)];

                let u_ijm1 = mat_n[get_idx(i, j - 1, stride_trans)];
                let u_ijp1 = mat_n[get_idx(i, j + 1, stride_trans)];
                
                let res_n = residual(u_ij, u_ip1j, u_im1j_n, u_ijp1, u_ijm1,
                                     dx_sq, dy_sq, 0.0);

                if res_n.abs() >= res_max {
                    res_max = res_n.abs();
                }
                
                let sj = -1.0 * inv_dx_sq * (u_ip1j + u_im1j_np1);

                let numerator = sj - pj * s_prime[j - 1];
                let denominator = qj - pj * r_prime[j - 1];
                s_prime[j] = numerator / denominator;
            }

            let mut u_ijp1_squiggle = bc2;
            for j in (1 .. rows - 1).rev() {

                u_ijp1_squiggle = update(mat_n, mat_np1,
                                         &s_prime, &r_prime, u_ijp1_squiggle,
                                         i, j, stride_trans);
            }
        }

        max_residuals.push(res_max);

        if res_max >= RES_MAX  { //continue iteration
            mem::swap(&mut mat_n, &mut mat_np1); // swap mat_n and mat_np1
            even_swap_count = !even_swap_count;
        } else {
            break;// finish iteration.
        }
    }

    if even_swap_count {
        transpose_back(matrix, mat_n, rows, columns);
    } else {
        transpose_back(matrix, mat_np1, rows, columns);
    }

    max_residuals
}

fn update(mat_n: &mut [f64], mat_np1: &mut [f64],
          s_prime: &[f64], r_prime: &[f64], u_ijp1_squiggle: f64,
          i: usize, j: usize, stride: usize) -> f64 {

    let u_ij_squiggle = s_prime[j] - r_prime[j] * u_ijp1_squiggle;
    let u_ij_n = mat_n[get_idx(i, j, stride)];
    let u_ij_np1 = u_ij_n + OMEGA * (u_ij_squiggle - u_ij_n);
    mat_np1[get_idx(i, j, stride)] = u_ij_np1;

    u_ij_squiggle
}

fn transpose_back(mat_orig: &mut [f64], mat_trans: &mut [f64],
                  rows: usize, columns: usize) {
    for row in 0 .. rows {
        for column in 0 .. columns {
            let orig_idx = get_idx(row, column, columns);
            let trans_idx = get_idx(column, row, rows);
            mat_orig[orig_idx] = mat_trans[trans_idx];
        }
    }
}
