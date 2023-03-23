__global__ void MatrixMulBlocksKernel(float *Md, float *Nd, float *Pd, int Width) {

	// Calculate the row index of the Pd element and M
	int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	// Calculate the column index of Pd and N
	int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	float Pvalue = 0;

	for (int k = 0; k < Width; ++k)
		Pvalue += Md[Row * Width + k] * Nd[k * Width + Col];

	Pd[Row * Width + Col] = Pvalue;
}

