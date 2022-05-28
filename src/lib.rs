use pyo3::prelude::*;

use std::env;
use std::fmt;
#[derive(PartialEq, Clone, Copy)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
}

impl DataPoint {
    pub fn new(x: f64, y: f64) -> Self {
        DataPoint { x, y }
    }
}

impl fmt::Debug for DataPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {:.1})", self.x, self.y)
    }
}

fn create_datapoints(x: &Vec<f64>, y: &Vec<f64>) -> Vec<DataPoint> {
    let mut data_points: Vec<DataPoint> = Vec::new();
    for i in 0..x.len() {
        data_points.push(DataPoint { x: x[i], y: y[i] });
    }
    data_points
}

fn data_points_to_vector(data_points: &Vec<DataPoint>) -> (Vec<f64>, Vec<f64>) {
    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    for i in 0..data_points.len() {
        x.push(data_points[i].x);
        y.push(data_points[i].y);
    }
    (x, y)
}

fn calculate_linear_regression_coefficients(bucket: &Vec<DataPoint>) -> (f64, f64) {
    let avg_x = bucket.iter().map(|x| x.x).sum::<f64>() / bucket.len() as f64;
    let avg_y = bucket.iter().map(|x| x.y).sum::<f64>() / bucket.len() as f64;

    let mut a_num = 0.0;
    let mut a_den = 0.0;

    for point in bucket.iter() {
        a_num += (point.x - avg_x) * (point.y - avg_y);
        a_den += (point.x - avg_x).powi(2);
    }

    let a = a_num / a_den;
    let b = avg_y - a * avg_x;
    (a, b)
}

fn calculate_sse_for_bucket(bucket: Vec<DataPoint>) -> f64 {
    let (a, b) = calculate_linear_regression_coefficients(&bucket);

    let sum_standard_error = bucket
        .iter()
        .map(|point| (point.y - (a * point.x + b)).powi(2))
        .sum::<f64>();

    sum_standard_error
}

fn calculate_sse_for_buckets(buckets: &Vec<Vec<DataPoint>>) -> Vec<f64> {
    let mut sse = vec![];

    for i in 1..buckets.len() - 1 {
        let previous_bucket = &buckets[i - 1];
        let current_bucket = &buckets[i];
        let next_bucket = &buckets[i + 1];

        let mut bucket_with_adjacent_points = vec![*previous_bucket.last().unwrap()];

        bucket_with_adjacent_points.extend(current_bucket.iter().clone());
        bucket_with_adjacent_points.push(*next_bucket.first().unwrap());

        sse.push(calculate_sse_for_bucket(
            bucket_with_adjacent_points.to_vec(),
        ));
    }

    sse
}

// Function to find the index of the largest value in a vector
fn find_largest_index(vec: &[f64], buckets: &Vec<Vec<DataPoint>>) -> usize {
    let mut max_sse = 0.0;
    let mut max_index = 0;

    for i in 0..vec.len() {
        if vec[i] > max_sse && buckets[i + 1].len() > 1 {
            max_index = i;
            max_sse = vec[i];
        }
    }

    max_index + 1
}

fn find_lowest_sse_adjacent_buckets(sse: &Vec<f64>, ignore_index: usize) -> usize {
    let mut min_sse_sum = f64::INFINITY;
    let mut min_sse_index = 0;

    for i in 0..sse.len() - 1 {
        if i == ignore_index - 1 || i == ignore_index {
            continue;
        }

        let sse_sum = sse[i] + sse[i + 1];
        if sse_sum < min_sse_sum {
            min_sse_sum = sse_sum;
            min_sse_index = i;
        }
    }
    min_sse_index + 1
}

fn split_bucket_at(mut buckets: Vec<Vec<DataPoint>>, index: usize) -> Vec<Vec<DataPoint>> {
    let bucket_length = buckets[index].len();
    if bucket_length < 2 {
        buckets
    } else {
        let bucket_left = buckets[index][0..bucket_length / 2].to_vec();
        let bucket_right = buckets[index][bucket_length / 2..bucket_length].to_vec();

        buckets.remove(index);
        buckets.insert(index, bucket_left);
        buckets.insert(index + 1, bucket_right);
        buckets
    }
}

fn merge_bucket_at(mut buckets: Vec<Vec<DataPoint>>, index: usize) -> Vec<Vec<DataPoint>> {
    let bucket_left = buckets[index].to_vec();
    let bucket_right = buckets[index + 1].to_vec();

    buckets.remove(index);
    buckets.remove(index);

    let mut merged_bucket = vec![];
    merged_bucket.extend(bucket_left);
    merged_bucket.extend(bucket_right);

    buckets.insert(index, merged_bucket);
    buckets
}

fn split_vec_into_buckets(
    vec: &[DataPoint],
    vector_length: usize,
    desired_length: usize,
) -> Vec<Vec<DataPoint>> {
    // Splits a vector into buckets of desired_length
    let mut split_data = vec![vec![vec[0]]];

    for chunk in vec[1..vector_length - 1]
        .to_vec()
        .chunks((((vector_length as f32) - 2.0) / ((desired_length as f32) - 2.0)).ceil() as usize)
    {
        split_data.push(chunk.to_vec());
    }

    split_data.push(vec![vec[vector_length - 1]]);
    split_data
}
pub fn lttb(data: &Vec<Vec<DataPoint>>) -> Vec<DataPoint> {
    // Select the point in the first bucket
    let mut datapoints = vec![data[0][0]];

    for i in 0..data.len() - 2 {
        let avg_x = data[i + 2].iter().map(|x| x.x).sum::<f64>() / data[i].len() as f64;
        let avg_y = data[i + 2].iter().map(|x| x.y).sum::<f64>() / data[i].len() as f64;

        let a = datapoints[i];
        let c = DataPoint { x: avg_x, y: avg_y };

        let mut max_area = 0.0;
        let mut next_point_index = 0;
        for (i, b) in data[i + 1].iter().enumerate() {
            let area = (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)).abs();
            if area > max_area {
                max_area = area;
                next_point_index = i;
            }
        }
        datapoints.push(data[i + 1][next_point_index]);
    }
    datapoints.push(data[data.len() - 1][0]);

    datapoints
}

#[pyfunction]
fn downsample(x: Vec<f64>, y: Vec<f64>, threshold: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    env::set_var("RUST_BACKTRACE", "1");

    let data_points = create_datapoints(&x, &y);
    let vector_length = data_points.len();
    if threshold == 0 {
        panic!("desired_length must be greater than 0");
    }

    if vector_length <= 2 || vector_length <= threshold {
        return Ok((x, y));
    }

    let mut buckets = split_vec_into_buckets(&data_points, vector_length, threshold);

    for _ in 0..data_points.len() * 10 / threshold {
        let sse = calculate_sse_for_buckets(&buckets);

        let highest_sse_index = find_largest_index(&sse, &buckets);

        if buckets[highest_sse_index].len() < 2 {
            println!("hej");
            break;
        }

        let mut lowest_sse_index = find_lowest_sse_adjacent_buckets(&sse, highest_sse_index);

        buckets = split_bucket_at(buckets, highest_sse_index);

        lowest_sse_index = if lowest_sse_index > highest_sse_index {
            lowest_sse_index + 1
        } else {
            lowest_sse_index
        };

        buckets = merge_bucket_at(buckets, lowest_sse_index);
    }

    let hej = lttb(&buckets);

    let (x, y) = data_points_to_vector(&hej);
    Ok((x, y))
}

/// A Python module implemented in Rust.
#[pymodule]
fn ltd_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(downsample, m)?)?;
    Ok(())
}
