#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	vector3* values;
	vector3** accels;

	cudaMallocManaged(values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMallocManaged(accels, sizeof(vector3*)*NUMENTITIES);

	//Parallelize the array population.								//I think this section is correct????
	__global__														//
	void populateArray(vector3* values, vector3** accels) {			//
																	//
		int i = blockIdx.x * blockDim.x + threadIdx.x;				//
		accels[i]=&values[i*NUMENTITIES];							//
																	//
	}

	int blockSize = 256;											//Blocksize can be messed around with
	int numBlocks = (NUMENTITIES + blockSize - 1) / blockSize;

	populateArray<<<numBlocks, blockSize>>>(values, accels);
	cudaDeviceSynchronize();



	__global__
	void calculate_pairwise_accelerations(vector3* values, vector3** accels){

		//figure out what element of the big 1-d array we are currently working on and
		//have each thread calculate exactly one index of data

		int currThreadId = blockId.x * blockDim.x + threadIdx.x;		//Hold the calculation so we don't have to repeat it twice (for a few extra nanoseconds :D)

		i = currThreadId / NUMENTITIES; 								//define i in terms of what block/dimension/thread we are currently using
		j = currThreadId % NUMENTITIES; 								//define j in terms of what block/dimension/thread we are currently using

		if(currThreadId < NUMENTITIES * NUMENTITIES) {					//Ensure that the result of our block is actually in bounds

			if(i == j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}

			else {
				vector3 distance;

				distance[0]=hPos[i][k]-hPos[j][k];
				distance[1]=hPos[i][k]-hPos[j][k];
				distance[2]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);

			}


		}
	
	
	}

	calculate_pairwise_accelerations<<<numBlocks, blockSize>>>(values, accels);
	cudaDeviceSynchronize();

	/**first compute the pairwise accelerations.  Effect is on the first argument.
	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
	**/

	__global__
	void update_velocity_and_position(vector3* values, vector3** accels){


		int currThreadId = blockId.x * blockDim.x + threadIdx.x;		//Hold the calculation so we don't have to repeat it twice (for a few extra nanoseconds :D)

		i = currThreadId / NUMENTITIES; 								//define i in terms of what block/dimension/thread we are currently using
		j = currThreadId % NUMENTITIES; 								//define j in terms of what block/dimension/thread we are currently using

		if(currThreadId < NUMENTITIES * NUMENTITIES) {					//Ensure that the result of our block is actually in bounds
			
			vector3 accel_sum = {accels[currThreadId][0], accels[currThreadId][1], accels[currThreadId][2]};

			hVel[i][0]+=accel_sum[0]*INTERVAL;
			hPos[i][0]=hVel[i][0]*INTERVAL;

			hVel[i][1]+=accel_sum[1]*INTERVAL;
			hPos[i][1]=hVel[i][1]*INTERVAL;

			hVel[i][2]+=accel_sum[2]*INTERVAL;
			hPos[i][2]=hVel[i][2]*INTERVAL;

		}

	}

	update_velocity_and_position<<<numBlocks, blockSize>>>(values, accels);
	cudaDeviceSynchronize();


	/**sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
				
				accel_sum[0]+=accels[i][j][0];
				accel_sum[1]+=accels[i][j][1];
				accel_sum[2]+=accels[i][j][2];
		
		}

		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		hVel[i][0]+=accel_sum[0]*INTERVAL;
		hPos[i][0]=hVel[i][0]*INTERVAL;

		hVel[i][1]+=accel_sum[1]*INTERVAL;
		hPos[i][1]=hVel[i][1]*INTERVAL;

		hVel[i][2]+=accel_sum[2]*INTERVAL;
		hPos[i][2]=hVel[i][2]*INTERVAL;
	}*/

	cudaFree(accels);
	cudaFree(values);
}
