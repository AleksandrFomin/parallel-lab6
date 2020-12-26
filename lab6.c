#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
 
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

struct timeval timeval;
FILE *fp;
char *source_str;
size_t source_size;
cl_command_queue command_queue;
cl_context context;
cl_device_id device_id = NULL;
cl_int ret;
cl_program program;

enum lab_stages {
	MAP = 0,
	MERGE,
	STAGES_AMOUNT,
};

long stages_time[STAGES_AMOUNT] = { 0 };

void print_stages_time()
{
	int i;
	for (i = 0; i < STAGES_AMOUNT; i++) {
		printf(" %ld", stages_time[i]);
	}

	printf("\n");
}

double omp_get_wtime()
{
	gettimeofday(&timeval, NULL);
	return (double)timeval.tv_sec + (double)timeval.tv_usec / 1000000;
}

void fill_array(double *arr, int size, double left, double right, unsigned int *seedp)
{	
	int i;

	for (i = 0; i < size; i++) {
		unsigned int seed_i = i + *seedp;
		arr[i] = rand_r(&seed_i) / (double)RAND_MAX * (right - left) + left;
	}
}

void print_array(double *arr, int size)
{
	int i;
	printf("arr=[");
	for (i = 0; i < size; i++) {
		printf(" %f", arr[i]);
	}
	printf("]\n");
}

void map_m1(double *arr, int size)
{
	// Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "map_m1_func", &ret);

	// Create memory buffers on the device for each vector 
    cl_mem arr_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
    									size * sizeof(double), NULL, &ret);
 
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, arr_mem_obj, CL_TRUE, 0,
    						   size * sizeof(double), arr, 0, NULL, NULL);
 
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&arr_mem_obj);
 
    // Execute the OpenCL kernel on the list
    size_t global_item_size = size; // Process the entire lists
    size_t local_item_size = 1; // Divide work items into groups of 64

    cl_event event;  // creating an event variable for timing

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, &event);

    ret = clWaitForEvents (1, &event); // Wait for the event

    ret = clEnqueueReadBuffer(command_queue, arr_mem_obj, CL_TRUE, 0,
    						  size * sizeof(double), arr, 0, NULL, NULL);

    // Obtain the start- and end time for the event
	unsigned long start = 0;
	unsigned long end = 0;

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
    						sizeof(cl_ulong), &start, NULL);       
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
    						sizeof(cl_ulong), &end, NULL);

	// Compute the duration in nanoseconds
	unsigned long duration = end - start;
	stages_time[MAP] += duration / 1000000;

	// Don't forget to release the vent
	clReleaseEvent(event);

	ret = clReleaseKernel(kernel);
	ret = clReleaseMemObject(arr_mem_obj);
}

void map_m2(double *arr, int size, double *arr_copy)
{
	// Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "map_m2_func", &ret);

	// Create memory buffers on the device for each vector 
    cl_mem arr_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
    									size * sizeof(double), NULL, &ret);
    cl_mem arr2_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
    									 size * sizeof(double), NULL, &ret);
 
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, arr_mem_obj, CL_TRUE, 0,
    						   size * sizeof(double), arr, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, arr2_mem_obj, CL_TRUE, 0,
    						   size * sizeof(double), arr_copy, 0, NULL, NULL);
 
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&arr_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&arr2_mem_obj);
 
    // Execute the OpenCL kernel on the list
    size_t global_item_size = size; // Process the entire lists
    size_t local_item_size = 1; // Divide work items into groups of 64

    cl_event event;  // creating an event variable for timing

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, &event);

    ret = clWaitForEvents (1, &event); // Wait for the event

    ret = clEnqueueReadBuffer(command_queue, arr_mem_obj, CL_TRUE, 0,
    						  size * sizeof(double), arr, 0, NULL, NULL);

    // Obtain the start- and end time for the event
	unsigned long start = 0;
	unsigned long end = 0;

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
    						sizeof(cl_ulong), &start, NULL);       
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
    						sizeof(cl_ulong), &end, NULL);

	// Compute the duration in nanoseconds
	unsigned long duration = end - start;
	stages_time[MAP] += duration / 1000000;

	// Don't forget to release the vent
	clReleaseEvent(event);

	ret = clReleaseKernel(kernel);
	ret = clReleaseMemObject(arr_mem_obj);
	ret = clReleaseMemObject(arr2_mem_obj);
}

void copy_arr(double *src, int len, double *dst)
{
	int i;

	for (i = 0; i < len; i++)
		dst[i] = src[i];
}

void apply_merge_func(double *m1, double *m2, int m2_len)
{
	// Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "merge_func", &ret);

	// Create memory buffers on the device for each vector 
    cl_mem arr_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
    									m2_len * sizeof(double), NULL, &ret);
    cl_mem arr2_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
    									 m2_len * sizeof(double), NULL, &ret);
 
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, arr_mem_obj, CL_TRUE, 0,
    						   m2_len * sizeof(double), m1, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, arr2_mem_obj, CL_TRUE, 0,
    						   m2_len * sizeof(double), m2, 0, NULL, NULL);
 
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&arr_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&arr2_mem_obj);
 
    // Execute the OpenCL kernel on the list
    size_t global_item_size = m2_len; // Process the entire lists
    size_t local_item_size = 1; // Divide work items into groups of 64

    cl_event event;  // creating an event variable for timing

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, &event);

    ret = clWaitForEvents (1, &event); // Wait for the event
    // printf("clWaitForEvents\n");

    ret = clEnqueueReadBuffer(command_queue, arr2_mem_obj, CL_TRUE, 0,
    						  m2_len * sizeof(double), m2, 0, NULL, NULL);

    // Obtain the start- and end time for the event
	unsigned long start = 0;
	unsigned long end = 0;

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
    						sizeof(cl_ulong), &start, NULL);       
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
    						sizeof(cl_ulong), &end, NULL);

	// Compute the duration in nanoseconds
	unsigned long duration = end - start;
	stages_time[MERGE] += duration / 1000000;

	// Don't forget to release the event
	clReleaseEvent(event);

	ret = clReleaseKernel(kernel);
	ret = clReleaseMemObject(arr_mem_obj);
	ret = clReleaseMemObject(arr2_mem_obj);
}

void heapify(double *array, int n)
{
	int i,j,k;
	double item;
	for(k=1 ; k<n ; k++) {
		item = array[k];
		i = k;
		j = (i-1)/2;
		while( (i>0) && (item>array[j]) ) {
			array[i] = array[j];
			i = j;
			j = (i-1)/2;
		}
		array[i] = item;
	}
}

void adjust(double *array, int n)
{
	int i,j;
	double item;

	j = 0;
	item = array[j];
	i = 2*j+1;

	while(i<=n-1) {
		if(i+1 <= n-1)
			if(array[i] < array[i+1])
				i++;
		if(item < array[i]) {
			array[j] = array[i];
			j = i;
			i = 2*j+1;
		} else
			break;
	}
	array[j] = item;
}

void heapsort(double *array, int n)
{
	int i;
	double t;

	heapify(array,n);

	for(i=n-1 ; i>0 ; i--) {
		t = array[0];
		array[0] = array[i];
		array[i] = t;
		adjust(array,i);
	}
}

void mergeArrays(double *dst, double *arr1, double *arr2, int len1, int len2)
{
	int i, j, k;
	i = j = k = 0;
	for (i = 0; i < len1 && j < len2;) {
		if (arr1[i] < arr2[j]) {
			dst[k] = arr1[i];
			k++;
			i++;
		} else {
			dst[k] = arr2[j];
			k++;
			j++;
		}
	}
	while (i < len1) {
		dst[k] = arr1[i];
		k++;
		i++;
	}
	while (j < len2) {
		dst[k] = arr2[j];
		k++;
		j++;
	}
}

double min_not_null(double *arr, int len)
{
	int i;
	double min_val = DBL_MAX;
	for (i = 0; i < len; i++) {
		if (arr[i] < min_val && arr[i] > 0)
			min_val = arr[i];
	}
	return min_val;
}

double reduce(double *arr, int len)
{
	int i;
	double min_val = min_not_null(arr, len);
	double x = 0;

	for (i = 0; i < len; i++) {
		if ((int)(arr[i] / min_val) % 2 == 0) {
			double sin_val = sin(arr[i]);
			x += sin_val;
		}
	}
	return x;
}

void do_main(int argc, char* argv[])
{
	int i, N, N2;
	double T1, T2;
	long delta_ms;
	double *M1, *M2, *M2_copy, *MERGED;
	int A = 540;
	unsigned int seed1, seed2;
	// double X;
	int iter = 50;

	N = atoi(argv[1]); /* N равен первому параметру командной строки */
	T1 = omp_get_wtime(); /* запомнить текущее время T1 */

	M1 = malloc(sizeof(double) * N);
	M2 = malloc(sizeof(double) * N / 2);
	M2_copy = malloc(sizeof(double) * N / 2);
	MERGED = malloc(sizeof(double) * N / 2);

	for (i = 0; i < iter; i++) /* 50 экспериментов */
	{	
		seed1 = i;
		seed2 = i;
		fill_array(M1, N, 1, A, &seed1);
		fill_array(M2, N / 2, A, 10 * A, &seed2);
		// print_array(M1, N);
		
		map_m1(M1, N);
		// print_array(M1, N);
		copy_arr(M2, N / 2, M2_copy);
		map_m2(M2, N / 2, M2_copy);

		apply_merge_func(M1, M2, N / 2);

		N2 = N / 2;

		// heapsort(M2, N2);
		// print_array(M2, N2);

		reduce(MERGED, N / 2);
		// printf("X = %f\n", X);
	}
	T2 = omp_get_wtime(); /* запомнить текущее время T2 */

	delta_ms = (T2 - T1) * 1000;
	printf("%d %ld", N, delta_ms); /* T2 - T1 */
	print_stages_time();

	free(M1);
	free(M2);
	free(M2_copy);
	free(MERGED);
}

int main(int argc, char* argv[])
{
	FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Get platform and device information
	cl_platform_id platform_id = NULL; 
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_CPU, 1, 
            &device_id, &ret_num_devices);
 
    // Create an OpenCL context
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 
    // Create a command queue
    command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    // Create a program from the kernel source
    program = clCreateProgramWithSource(context, 1,
            (const char **)&source_str, (const size_t *)&source_size, &ret);
 
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	do_main(argc, argv);

	ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseProgram(program);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

	return 0;
}
