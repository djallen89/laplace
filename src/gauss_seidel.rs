use std::mem;
use super::{get_idx, residual, update};
use super::RES_MAX;

pub fn gauss_seidel(matrix: &mut Vec<f64>, rows: usize, columns: usize,
                    dx: f64, dy: f64) -> Vec<f64> {

    let mut mat_new = matrix.clone(); // copy data of original
    let mut mat_n = matrix; // copy the pointer; 
    let mut mat_np1 = &mut mat_new;

    let mut max_residuals = Vec::with_capacity(1000); // 1000 is a guess for n.

    let dx_sq = dx * dx;
    let dy_sq = dy * dy;
    let stride = columns; // stride is the number of columns per row

    let mut even_swap_count = true;

    loop {
        let mut res_max = 0.0f64;
        
        for row in 1 .. rows - 1 {
            for column in 1 .. columns - 1 {
                let u_ij_n = mat_n[get_idx(row, column, stride)];
                
                let u_ip1j_n = mat_n[get_idx(row, column + 1, stride)];
                let u_im1j_n = mat_n[get_idx(row, column - 1, stride)];
                let u_im1j_np1 = mat_np1[get_idx(row, column - 1, stride)];

                let u_ijp1_n = mat_n[get_idx(row + 1, column, stride)];
                let u_ijm1_n = mat_n[get_idx(row - 1, column, stride)];
                let u_ijm1_np1 = mat_np1[get_idx(row - 1, column, stride)];

                let res_n = residual(u_ij_n, u_ip1j_n, u_im1j_n, u_ijp1_n, u_ijm1_n,
                                   dx_sq, dy_sq, 0.0);

                if res_n.abs() >= res_max {
                    res_max = res_n;
                }
                
                let u_ij_np1 = update(u_ip1j_n, u_im1j_np1, u_ijp1_n, u_ijm1_np1,
                                      dx_sq, dy_sq);
                mat_np1[get_idx(row, column, stride)] = u_ij_np1;
            }
        }

        max_residuals.push(res_max);
        
        if res_max >= RES_MAX { //continue iteration
            mem::swap(&mut mat_n, &mut mat_np1); // swap mat_n and mat_np1
            even_swap_count = !even_swap_count;
        } else {
            break // finish iteration.
        }
    }

    if even_swap_count { //mat_np1 points to mat_new
        *mat_n = mat_np1.clone();
    }

    max_residuals
}
