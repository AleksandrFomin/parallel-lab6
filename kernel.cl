__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    C[i] = A[i] + B[i];
}

__kernel void map_m1_func(__global double *arr) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    arr[i] = tanh(arr[i]) - 1;
}

__kernel void map_m2_func(__global double *arr, __global double *arr_copy) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    double prev = 0;
	if (i > 0)
		prev = arr_copy[i - 1];
	arr[i] = sqrt(exp(1.0) * (arr_copy[i] + prev));
}

__kernel void merge_func(__global double *m1, __global double *m2) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 	
    // Do the operation
    m2[i] = fabs(m1[i] - m2[i]);
}