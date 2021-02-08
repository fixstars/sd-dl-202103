/*
Copyright 2021 Fixstars Corporation
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http ://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>


static constexpr int NUM_TRIALS = 11;


// CPU版ReLUカーネル
void relu_cpu(float *dst, const float *src, int n){
	for(int i = 0; i < n; ++i){
		dst[i] = std::max(0.0f, src[i]);
	}
}

// CPU版処理時間計測
// NUM_TRIALS 回計測して中央値を求める
double relu_cpu_benchmark(float *dst, const float *src, int n){
	std::vector<double> durations(NUM_TRIALS);
	for(int i = 0; i < NUM_TRIALS; ++i){
		const auto begin = std::chrono::steady_clock::now();
		relu_cpu(dst, src, n);
		const auto end = std::chrono::steady_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
		durations[i] = duration.count() * 1e-3;
	}
	std::sort(durations.begin(), durations.end());
	return durations[NUM_TRIALS / 2];
}


// GPU版ReLUカーネル
__global__ void relu_gpu(float *dst, const float *src, int n){
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){ dst[i] = max(0.0f, src[i]); }
}

void call_relu_gpu(float *dst, const float *src, int n){
	const int bdim = 256, gdim = (n + bdim - 1) / bdim;
	relu_gpu<<<gdim, bdim>>>(dst, src, n);
}

// GPU版処理時間計測
// NUM_TRIALS 回計測して中央値を求める
double relu_gpu_benchmark(float *h_dst, const float *h_src, int n){
	// デバイスメモリの確保
	float *d_dst = nullptr, *d_src = nullptr;
	cudaMalloc(&d_src, sizeof(float) * n);
	cudaMalloc(&d_dst, sizeof(float) * n);
	// 入力データの転送
	cudaMemcpy(d_src, h_src, sizeof(float) * n, cudaMemcpyHostToDevice);

	std::vector<double> durations(NUM_TRIALS);
	for(int i = 0; i < NUM_TRIALS; ++i){
		const auto begin = std::chrono::steady_clock::now();
		call_relu_gpu(d_dst, d_src, n);
		cudaDeviceSynchronize();  // GPUカーネルの終了を待つ
		const auto end = std::chrono::steady_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
		durations[i] = duration.count() * 1e-3;
	}

	// 出力データの転送
	cudaMemcpy(h_dst, d_dst, sizeof(float) * n, cudaMemcpyDeviceToHost);
	// デバイスメモリの開放
	cudaFree(d_src);
	cudaFree(d_dst);

	// 中央値を求める
	std::sort(durations.begin(), durations.end());
	return durations[NUM_TRIALS / 2];
}


// 検算
bool validate(const float *expect, const float *actual, int n){
	bool valid = true;
	for(int i = 0; i < n; ++i){
		if(std::fabs(expect[i] - actual[i]) > 1e-4){
			valid = false;
		}
	}
	return valid;
}


int main(int argc, char *argv[]){
	if(argc < 2){
		std::cerr << "Usage: " << argv[0] << " n" << std::endl;
		return 0;
	}

	const int n = atoi(argv[1]);
	std::cout << "n = " << n << std::endl;

	std::default_random_engine engine;
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<float> src(n), cpu_dst(n), gpu_dst(n);
	for(int i = 0; i < n; ++i){
		src[i] = dist(engine);
	}

	const auto cpu_duration =
		relu_cpu_benchmark(cpu_dst.data(), src.data(), n);
	std::cout << "CPU: " << cpu_duration << " [ms]" << std::endl;

	const auto gpu_duration =
		relu_gpu_benchmark(gpu_dst.data(), src.data(), n);
	std::cout << "GPU: " << gpu_duration << " [ms]" << std::endl;

	const auto valid = validate(cpu_dst.data(), gpu_dst.data(), n);
	std::cout << "Validation: " << (valid ? "Success" : "Failed") << std::endl;

	return 0;
}
