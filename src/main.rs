pub use numerical::NumerMethod;

use std::path::Path;
use std::fs::{File, create_dir};
use std::io::Write;
use std::error::Error;
use exact::exact;
use jacobi::jacobi;
use jacobi::_jacobi;
use gauss_seidel::gauss_seidel;
use over_relax::over_relax;
use numerical::numerical;

mod exact;
mod jacobi;
mod gauss_seidel;
mod over_relax;
mod numerical;

pub const L: f64 = 1.0;
pub const W: f64 = 1.0;
pub const RES_MAX: f64 = 1.0e-5;

pub fn get_idx(row: usize, column: usize, stride: usize) -> usize {
    row * stride + column
}

pub fn create_phi_matrix(rows: usize, columns: usize) -> Vec<f64> {
    let mut phi_matrix = Vec::with_capacity(rows * columns);
    for _row in 0 .. rows - 1 {
        for _column in 0 .. columns {
            phi_matrix.push(0.0);
        }
    }

    phi_matrix.push(0.0);
    for _column in 1 .. columns - 1 {
        phi_matrix.push(1.0);
    }
    phi_matrix.push(0.0);
    
    phi_matrix
}

/// row is j, column is i, stride is the number of columns per row
pub fn residual(u_ij: f64, u_ip1j: f64, u_im1j: f64, u_ijp1: f64, u_ijm1: f64,
                dx_sq: f64, dy_sq: f64, fij: f64) -> f64 {
    let u_ij_2 = 2.0 * u_ij;
    
    let lhterm = (u_ip1j - u_ij_2 + u_im1j) / dx_sq;
    let rhterm = (u_ijp1 - u_ij_2 + u_ijm1) / dy_sq;

    lhterm + rhterm - fij
}

pub fn update(u_ip1j: f64, u_im1j: f64, u_ijp1: f64, u_ijm1: f64,
          dx_sq: f64, dy_sq: f64) -> f64 {

    let numer = dy_sq * (u_ip1j + u_im1j) + dx_sq * (u_ijp1 + u_ijm1);
    let denom = 2.0 * (dy_sq + dx_sq);
    let u_ij_np1 = numer / denom;
    u_ij_np1
}

fn main() {
    setup_output_dir();
    do_exact();
    do_func(50, 50, &_jacobi, "jacobi");
    do_func(100, 100, &_jacobi, "jacobi");
    //do_method(50, 50, jacobi, "jacobi");
    //do_method(100, 100, jacobi, "jacobi");
    do_func(50, 50, &gauss_seidel, "gauss_seidel");
    do_func(100, 100, &gauss_seidel, "gauss_seidel");
    do_func(50, 50, &over_relax, "over_relaxation");
    do_func(100, 100, &over_relax, "over_relaxation");
}

fn do_exact() {
    let rows = 200;
    let cols = 200;
    let n = 50;
    let mut grid = Vec::with_capacity(rows * cols);
    for _i in 0 .. rows * cols {
        grid.push(0.0);
    }
    
    exact(&mut grid, rows, cols, n);

    let filename = format!("exact{}", n);
    write_arr(&grid, rows, cols, &filename);
    let whalf_begin = rows / 2 * cols;
    let whalf_end = whalf_begin + cols;
    let dx = L / ((cols - 1) as f64);
    write_vector(&grid[whalf_begin .. whalf_end], dx, "x", "phi", "exact_whalf");
}
/*
fn do_func(rows: usize, columns: usize,
           func: &Fn(&mut Vec<f64>, usize, usize, f64, f64) -> Vec<f64>,
           name: &str) {
 */
fn do_func(rows: usize, columns: usize,
           func: &Fn(&mut Vec<f64>, usize, usize, f64, f64) -> Vec<f64>,
           name: &str) {
    let mut matrix = create_phi_matrix(rows, columns);

    let dx = L / ((columns - 1) as f64);
    let dy = W / ((rows - 1) as f64);
    let n_arr = func(&mut matrix, rows, columns, dx, dy);
    let whalf_begin = rows / 2 * columns;
    let whalf_end = whalf_begin + columns;
    
    let filename = format!("{}{}", name, rows); // assume rows = columns
    write_arr(&matrix, rows, columns, &filename);

    let vec_name = format!("{}_whalf{}", name, rows);
    write_vector(&matrix[whalf_begin .. whalf_end], dx, "x", "phi", &vec_name);

    let n_name = format!("{}_n{}", name, rows);
    write_vector(&n_arr, 1.0, "n", "Res", &n_name);
}

fn do_method(rows: usize, columns: usize,
             method: NumerMethod, name: &str) {

    let mut matrix = create_phi_matrix(rows, columns);

    let dx = L / ((columns - 1) as f64);
    let dy = W / ((rows - 1) as f64);
    let n_arr = numerical(&mut matrix, rows, columns, dx, dy, method);
    let whalf_begin = rows / 2 * columns;
    let whalf_end = whalf_begin + columns;
    
    let filename = format!("{}{}", name, rows); // assume rows = columns
    write_arr(&matrix, rows, columns, &filename);

    let vec_name = format!("{}_whalf{}", name, rows);
    write_vector(&matrix[whalf_begin .. whalf_end], dx, "x", "phi", &vec_name);

    let n_name = format!("{}_n{}", name, rows);
    write_vector(&n_arr, 1.0, "n", "Res", &n_name);
}

fn setup_output_dir() {
    let path = Path::new("./data/");
    if !path.exists() {
        match create_dir(&path) {
            Ok(_) => {},
            Err(reason) => panic!("{}", reason.description())
        }
    }
}

fn get_file(filename: &str, extension: &str) -> File {
    let pathname = format!("./data/{}.{}", filename, extension);
    let path = Path::new(&pathname);
    let file = match File::create(path) {
        Ok(f) => f,
        Err(reason) => panic!("{}", reason.description())
    };
    file
}

fn write_line(file: &mut File, line: &str) {
    match file.write_all(line.as_bytes()) {
        Ok(_) => {}, // success,
        Err(reason) => panic!("{}", reason.description())
    }    
}

fn write_arr(arr: &[f64], rows: usize, columns: usize, filename: &str) {
    let mut file = get_file(filename, "dat");
    
    for row in 0 .. rows {
        let mut row_string = "".to_string();
        for column in 0 .. columns - 1 {
            let entry = format!("{}\t", arr[get_idx(row, column, columns)]);
            row_string.push_str(&entry)
        }
        let entry = format!("{}\n", arr[get_idx(row, columns - 1, columns)]);
        row_string.push_str(&entry);
        write_line(&mut file, &row_string);
    }
}

fn write_vector(vec: &[f64], dx: f64, lcol: &str, rcol: &str, filename: &str) {
    let mut file = get_file(filename, "csv");

    let mut vec_string = format!("{}, {}\n", lcol, rcol);
    for (xi, entry) in vec.iter().enumerate() {
        let entry_string = format!("{}, {}\n", (xi as f64) * dx, entry);
        vec_string.push_str(&entry_string);
    }

    write_line(&mut file, &vec_string);
}

