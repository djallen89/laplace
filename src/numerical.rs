use std::mem; // mem is memory
use super::get_idx;
use super::residual;
use super::update;
use super::RES_MAX;

pub type NumerMethod = fn(usize, usize, usize, // row, column, stride
                          f64, f64, // dx_sq, dy_sq
                          &mut [f64], &mut [f64], &mut f64); //mat_n, mat_np1, max_res

pub fn numerical(matrix: &mut Vec<f64>, rows: usize, columns: usize,
                 dx: f64, dy: f64, method: NumerMethod) -> Vec<f64> {
    
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
                method(row, column, stride,
                       dx_sq, dy_sq,
                       mat_n, mat_np1, &mut res_max);
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
