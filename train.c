#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "NuMicro.h"

#define PLL_CLOCK       50000000

/******************************************************************
 * dataset format setting
 ******************************************************************/

#define train_data_num 120			//Total number of training data
#define test_data_num 24		//Total number of testing data

/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/
#define input_length 15				//The number of input
#define HiddenNodes 30						//The number of neurons in hidden layer
#define target_num 3						//The number of output

const float LearningRate = 0.001;						//Learning Rate
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float goal_acc = 	0.99;							//Target accuracy

// Create training dataset/output
float train_data_input[train_data_num][input_length] = {
{182,145,398,509,404,891,533,380,849,486,350,771,170,116,342},

{180,143,399,556,444,952,526,379,849,292,199,508,171,118,347},

{182,144,400,461,364,814,548,405,899,218,146,414,168,116,341},

{182,143,396,494,396,870,520,383,856,229,152,429,168,116,341},

{180,143,392,559,476,998,582,413,913,256,170,467,169,116,340},

{184,145,398,413,319,733,469,320,770,265,175,465,170,117,343},

{185,145,401,417,332,752,448,320,749,288,194,501,170,118,345},

{183,143,397,511,411,896,543,395,881,238,159,441,168,115,338},

{192,146,401,467,366,809,519,370,825,234,155,430,170,116,340},

{184,147,399,546,453,962,573,413,907,302,203,522,168,116,340},

{185,146,399,472,316,740,589,467,996,227,152,430,169,119,344},

{182,141,397,419,299,701,511,387,863,340,237,580,172,118,348},

{184,144,398,420,294,696,540,434,937,221,149,421,170,118,344},

{186,143,394,390,268,648,554,450,976,236,161,442,169,118,341},

{191,144,399,360,243,601,555,442,953,264,178,477,171,120,348},

{192,145,400,333,230,575,445,309,731,382,278,648,171,119,346},

{184,144,402,467,318,743,576,455,968,217,149,417,170,118,345},

{188,145,400,448,305,704,592,477,1010,225,154,432,172,118,346},

{187,145,397,433,286,682,559,431,939,243,164,451,172,120,348},

{187,146,399,468,312,729,591,482,1016,206,141,405,169,117,342},

{139,348,395,448,357,751,592,492,978,211,155,389,188,139,348},

{192,161,393,598,477,953,516,384,780,192,145,368,185,137,346},

{137,345,398,603,473,946,548,416,832,192,133,369,185,137,345},

{192,163,397,621,465,941,558,438,881,192,146,370,187,137,347},

{146,364,393,142,352,1078,192,147,367,192,143,353,192,141,348},

{214,163,393,627,489,964,535,405,816,192,145,370,185,137,347},

{192,161,393,605,485,966,375,276,610,192,143,367,185,137,347},

{192,164,398,614,484,960,639,502,960,192,146,374,187,138,348},

{206,165,401,626,473,945,350,253,574,192,144,367,186,137,346},

{192,162,393,608,482,955,544,421,861,192,145,370,186,137,346},

{177,144,396,575,475,989,529,382,859,192,139,401,167,117,339},

{180,145,398,576,491,1019,557,393,880,259,170,466,172,119,348},

{185,146,400,514,432,930,541,388,871,253,169,448,168,116,339},

{184,145,398,538,434,925,552,391,876,234,153,436,168,117,342},

{182,146,397,605,504,1039,560,394,883,192,139,403,169,117,341},

{183,146,401,428,354,794,532,404,890,251,169,466,169,118,346},

{183,145,400,549,445,942,546,385,862,230,152,434,170,118,345},

{183,145,396,562,462,972,557,390,875,270,178,480,168,117,341},

{183,146,400,559,448,960,563,395,884,232,155,436,171,118,346},

{183,146,399,576,448,960,559,384,864,238,160,448,167,116,341},

{180,144,397,447,395,866,489,402,881,296,192,518,167,116,340},

{116,342,395,387,338,771,478,391,858,543,523,1151,169,116,342},

{177,143,399,354,294,698,445,375,833,326,235,572,170,119,345},

{181,146,399,563,507,1040,505,418,913,243,168,459,168,117,340},

{119,349,400,475,418,915,521,436,936,459,370,808,166,117,340},

{179,144,401,424,360,804,448,390,861,226,156,432,168,118,342},

{170,138,391,365,306,719,491,404,882,429,332,776,169,118,345},

{176,144,395,514,446,946,504,423,915,192,141,407,166,117,341},

{174,143,401,573,498,1024,531,444,951,192,142,407,167,117,341},

{179,143,398,455,386,843,466,383,841,311,219,544,168,119,346},

{177,144,400,477,390,850,558,485,1010,210,146,416,166,116,340},

{182,147,400,369,292,683,542,470,990,275,191,496,169,118,345},

{179,142,398,362,284,669,513,432,929,229,157,432,168,117,342},

{182,145,401,396,313,722,538,465,979,222,153,429,166,117,343},

{180,145,400,496,403,874,575,495,1017,203,141,405,168,117,341},

{177,141,394,406,320,741,513,444,945,232,160,443,169,118,344},

{177,143,401,391,309,716,521,437,938,208,145,410,165,116,339},

{183,149,401,405,320,743,536,464,980,279,194,503,169,118,346},

{181,144,393,405,325,744,538,460,972,271,188,491,168,119,344},

{182,141,396,414,320,740,448,373,818,232,158,432,166,117,339},

{192,166,400,560,484,943,549,448,907,192,147,371,185,138,345},

{192,165,396,566,480,936,543,459,905,192,147,375,185,136,348},

{192,166,401,550,456,912,611,540,1041,192,150,383,186,138,348},

{143,363,393,410,371,776,532,466,930,205,153,392,186,137,347},

{145,370,392,576,490,950,648,596,1129,210,154,389,185,137,347},

{192,165,400,541,448,899,440,336,699,192,146,373,187,138,347},

{178,147,400,451,388,836,493,423,911,253,179,477,171,119,345},

{118,341,396,528,447,934,448,377,832,341,246,600,167,118,341},

{180,147,398,453,397,859,466,395,863,290,207,524,169,119,345},

{152,430,399,396,319,745,438,359,802,514,442,941,169,119,347},

{174,145,400,526,454,952,462,385,844,253,175,467,171,121,347},

{179,146,395,493,454,941,446,349,782,244,167,460,168,118,343},

{133,383,395,428,402,874,539,448,957,455,375,827,169,120,347},

{181,148,400,515,446,924,463,389,850,211,147,412,168,119,341},

{170,145,397,307,284,675,453,373,807,435,347,749,167,119,339},

{180,148,400,506,448,956,464,362,798,366,263,611,169,118,344},

{177,145,395,527,474,982,459,372,835,238,167,453,172,120,348},

{180,146,399,402,338,758,400,319,739,225,157,430,172,119,339},

{180,146,398,482,416,891,411,320,753,229,159,433,167,118,342},

{179,144,393,540,509,1039,461,369,818,254,178,476,167,120,343},

{178,145,402,448,424,905,415,365,823,248,177,465,167,118,341},

{178,147,401,438,403,866,421,375,839,242,172,461,167,118,339},

{176,145,400,436,401,865,415,366,821,264,178,488,167,119,345},

{176,144,400,437,398,861,419,368,823,270,195,501,168,120,347},

{175,145,401,448,421,897,421,376,832,214,151,421,165,117,338},

{173,143,400,466,428,913,433,387,861,210,147,415,165,116,340},

{174,145,398,481,439,928,427,380,851,243,176,466,169,120,346},

{174,144,395,436,398,860,424,374,832,225,160,436,166,118,342},

{176,145,400,425,388,847,419,370,834,260,178,484,169,119,347},

{175,144,395,496,448,958,429,372,832,239,166,461,166,117,340},

{176,145,401,379,320,740,448,428,923,217,153,426,165,117,339},

{175,144,401,302,253,617,431,401,875,269,192,492,167,119,344},

{174,144,400,317,271,646,387,347,790,353,269,627,168,119,345},

{175,144,397,320,277,658,374,336,773,301,217,527,166,118,342},

{174,144,399,415,365,805,426,386,861,375,293,662,168,118,347},

{175,143,397,308,258,628,448,428,918,227,161,437,165,117,341},

{178,143,401,291,243,598,402,373,831,314,229,555,169,120,347},

{173,140,393,365,307,704,442,412,897,251,177,469,166,119,347},

{174,142,394,339,283,672,412,381,849,265,186,485,166,117,342},

{176,144,397,354,299,697,398,358,809,317,241,582,165,117,340},

{192,163,395,569,512,1002,611,576,1109,192,149,379,184,137,348},

{192,165,401,537,482,952,483,422,847,192,147,377,184,137,348},

{192,165,400,481,427,868,440,381,778,191,145,369,184,132,345},

{192,163,397,530,476,946,413,348,723,191,145,370,183,137,346},

{192,164,399,540,482,954,546,504,996,205,155,395,186,138,344},

{192,165,398,556,500,984,517,453,893,192,146,373,185,137,347},

{179,147,400,500,441,936,521,494,1036,192,141,405,165,116,338},

{177,145,397,448,396,865,448,404,894,240,175,463,165,117,339},

{176,145,398,440,383,842,497,465,984,214,153,425,167,118,344},

{178,147,397,475,425,910,448,408,899,266,193,499,165,117,340},

{177,148,401,445,405,871,420,388,855,233,169,449,166,118,341},

{177,148,401,448,404,865,448,408,884,214,152,420,167,117,340},

{177,148,401,436,395,852,434,395,861,228,160,431,168,119,342},

{177,146,397,465,427,904,424,383,832,213,152,417,166,118,340},

{176,147,401,476,431,912,421,381,832,247,174,448,168,118,345},

{175,147,399,503,456,950,460,416,896,188,140,396,166,118,340},

{174,147,401,496,446,944,477,432,935,189,141,399,167,118,340},

{177,145,401,414,372,819,433,393,868,239,166,455,168,120,345},

{174,143,393,465,429,913,415,371,834,275,203,508,169,120,348},

{176,145,401,399,355,786,401,362,817,308,229,552,170,121,348}
};
/*

*/
int train_data_output[train_data_num][target_num]  = {
{1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},
{1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},
{1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},
{1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},
{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}
};
	

float test_data_input[test_data_num][input_length] = {
{185,146,400,531,425,924,556,396,886,237,160,444,168,117,340},

{182,144,397,555,448,960,557,391,878,250,165,448,171,117,346},

{186,144,392,440,298,704,590,471,1009,247,167,457,171,118,346},

{185,144,400,506,344,788,600,478,1011,203,139,403,170,118,344},

{192,163,398,599,484,960,457,349,730,192,144,370,187,137,347},

{192,162,395,622,495,982,378,269,607,192,143,366,186,136,347},

{180,142,393,543,436,932,546,390,875,249,165,456,172,119,348},

{227,183,401,340,251,557,354,265,573,201,165,409,175,148,348},

{179,144,395,530,448,952,498,411,899,222,162,442,167,118,341},

{179,144,396,538,479,988,503,418,913,221,154,431,166,117,341},

{180,143,401,437,349,780,549,480,1001,210,146,414,167,117,342},

{180,145,401,441,359,800,542,466,979,257,177,478,170,120,348},

{180,147,401,496,418,887,521,454,960,248,177,471,168,118,345},

{180,147,398,491,425,903,510,442,944,276,193,498,170,119,347},

{179,145,398,463,436,939,560,471,978,453,365,805,171,120,348},

{178,146,401,468,408,870,439,363,814,248,177,466,170,120,348},

{176,146,401,479,443,943,432,380,849,258,179,482,168,119,348},

{172,143,400,269,257,642,381,338,769,360,326,693,165,117,339},

{173,141,396,320,271,650,438,403,878,231,162,432,164,117,341},

{174,144,400,381,326,749,475,440,942,236,168,451,166,117,344},

{174,143,393,475,415,892,472,430,928,206,147,415,165,117,341},

{178,147,401,495,438,934,474,430,939,242,179,475,166,118,341},

{172,147,401,448,403,875,409,371,830,292,218,534,169,121,348},

{174,144,394,483,429,914,488,434,927,188,137,394,165,116,340}
};

int test_data_output[test_data_num][target_num] = {
{1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},
{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, 
{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}
};

/******************************************************************
 * End Network Configuration
 ******************************************************************/


int ReportEvery10;
int RandomizedIndex[train_data_num];
long  TrainingCycle;
float Rando;
float Error;
float Accum;

float data_mean[15] ={0};
float data_std[15] ={0};

float Hidden[HiddenNodes];
float Output[target_num];
float HiddenWeights[input_length+1][HiddenNodes];
float OutputWeights[HiddenNodes+1][target_num];
float HiddenDelta[HiddenNodes];
float OutputDelta[target_num];
float ChangeHiddenWeights[input_length+1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes+1][target_num];

int target_value;
int out_value;
int max;


void scale_data()
{
		float sum[15] = {0};
		int i, j;
		
		// Compute Data Mean
		for(i = 0; i < train_data_num; i++){
			for(j = 0; j < input_length; j++){
				sum[j] += train_data_input[i][j];
			}
		}
		for(j = 0; j < input_length ; j++){
			data_mean[j] = sum[j] / train_data_num;
			printf("MEAN: %.2f\n", data_mean[j]);
			sum[j] = 0.0;
		}
		
		// Compute Data STD
		for(i = 0; i < train_data_num; i++){
			for(j = 0; j < input_length ; j++){
				sum[j] += pow(train_data_input[i][j] - data_mean[j], 2);
			}
		}
		for(j = 0; j < input_length; j++){
			data_std[j] = sqrt(sum[j]/train_data_num);
			printf("STD: %.2f\n", data_std[j]);
			sum[j] = 0.0;
		}
}

void normalize(float *data)
{
		int i;
	
		for(i = 0; i < input_length; i++){
			data[i] = (data[i] - data_mean[i]) / data_std[i];
		}
}

int train_preprocess()
{
    int i;
    
    for(i = 0 ; i < train_data_num ; i++)
    {
        normalize(train_data_input[i]);
    }
		
    return 0;
}

int test_preprocess()
{
    int i;

    for(i = 0 ; i < test_data_num ; i++)
    {
        normalize(test_data_input[i]);
    }
		
    return 0;
}

int data_setup()
{
    int i;
		//int j;
		int p, ret;
		unsigned int seed = 1;
	
		seed *= 1000;
		printf("\nRandom seed: %d\n", seed);
    srand(seed);

    ReportEvery10 = 1;
    for( p = 0 ; p < train_data_num ; p++ ) 
    {    
        RandomizedIndex[p] = p ;
    }
		
	scale_data();
    ret = train_preprocess();
    ret |= test_preprocess();
    if(ret) //Error Check
        return 1;

    return 0;
}

void run_train_data()
{
    int i, j, p;
    int correct=0;
    float accuracy = 0;
    printf("Train result:\n");
    for( p = 0 ; p < train_data_num ; p++ )
    { 
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (train_data_output[p][i] > train_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        
    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;

        if(out_value!=target_value)
            printf("Error --> Training Pattern: %d,Target : %d, Output : %d\n", p, target_value, out_value);
        else
            correct++;
        }
        // Calculate accuracy
        accuracy = (float)correct / train_data_num;
        printf ("Accuracy = %.2f /100 \n",accuracy*100);
}

void run_test_data()
{
    int i, j, p;
    int correct=0;
    float accuracy = 0;
    printf("Test result:\n");
    for( p = 0 ; p < test_data_num ; p++ )
    { 
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (test_data_output[p][i] > test_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        
    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += test_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;

        if(out_value!=target_value)
            printf("Error --> Training Pattern: %d,Target : %d, Output : %d\n", p, target_value, out_value);
        else
            correct++;
        }
        // Calculate accuracy
        accuracy = (float)correct / test_data_num;
        printf ("Accuracy = %.2f /100 \n",accuracy*100);
}

float Get_Train_Accuracy()
{
    int i, j, p;
    int correct = 0;
		float accuracy = 0;
    for (p = 0; p < train_data_num; p++)
    {
/******************************************************************
* Compute hidden layer activations
******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

/******************************************************************
* Compute output layer activations
******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        //get target value
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (train_data_output[p][i] > train_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        //get output value
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;
        //compare output and target
        if (out_value==target_value)
        {
            correct++;
        }
    }

    // Calculate accuracy
    accuracy = (float)correct / train_data_num;
    return accuracy;
}

void load_weight()
{
    int i,j;
    printf("\n=======Hidden Weight=======\n\n\n\n\n\n\n");
    printf("{");
    for(i = 0; i <= input_length ; i++)
    {
        printf("{");
        for (j = 0; j < HiddenNodes; j++)
        {
            if(j!=HiddenNodes-1){
                printf("%f,", HiddenWeights[i][j]);
            }else{
                printf("%f", HiddenWeights[i][j]);
            }
        }
        if(i!=input_length){
            printf("},\n");
        }else {
            printf("}");
        }
    }
    printf("}\n");

    printf("\n=======Output Weight=======\n");

    for(i = 0; i <= HiddenNodes ; i++)
    {
        printf("{");
        for (j = 0; j < target_num; j++)
        {
            if(j!=target_num-1){
                printf("%f,", OutputWeights[i][j]);
            }else{
                printf("%f", OutputWeights[i][j]);
            }
        }
        if(i!=HiddenNodes){
            printf("},\n");
        }else {
            printf("}");
        }
    }
    printf("}\n");
}

/*---------------------------------------------------------------------------------------------------------*/
/* MAIN function                                                                                           */
/*---------------------------------------------------------------------------------------------------------*/

int main(void)
{
	int i, j, p, q, r;
    float accuracy=0;

    // /* Unlock protected registers */
    // SYS_UnlockReg();

    // /* Init System, IP clock and multi-function I/O */
    // SYS_Init();

    // /* Lock protected registers */
    // SYS_LockReg();

    // /* Init UART0 for printf */
    // UART0_Init();
	
	//   GPIO_SetMode(PB, BIT2, GPIO_MODE_OUTPUT);
	//   PB2=0;
	
    printf("\n+-----------------------------------------------------------------------+\n");
    printf("|                        LAB8 - Machine Learning                        |\n");
    printf("+-----------------------------------------------------------------------+\n");

    printf("\n[Phase 1] Initialize DataSet ...");
	  /* Data Init (Input / Output Preprocess) */
		if(data_setup()){
        printf("[Error] Datasets Setup Error\n");
        return 0;
    }else
				printf("Done!\n\n");
		
		printf("[Phase 2] Start Model Training ...\n");
		// Initialize HiddenWeights and ChangeHiddenWeights 
    for( i = 0 ; i < HiddenNodes ; i++ ) {    
        for( j = 0 ; j <= input_length ; j++ ) { 
            ChangeHiddenWeights[j][i] = 0.0 ;
            Rando = (float)((rand() % 100))/100;
            HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    // Initialize OutputWeights and ChangeOutputWeights
    for( i = 0 ; i < target_num ; i ++ ) {    
        for( j = 0 ; j <= HiddenNodes ; j++ ) {
            ChangeOutputWeights[j][i] = 0.0 ;  
            Rando = (float)((rand() % 100))/100;        
            OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    // Begin training 
    for(TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++)
    {
        Error = 0.0 ;

        // Randomize order of training patterns
        for( p = 0 ; p < train_data_num ; p++) {
            q = rand()%train_data_num;
            r = RandomizedIndex[p] ; 
            RandomizedIndex[p] = RandomizedIndex[q] ; 
            RandomizedIndex[q] = r ;
        }

        // Cycle through each training pattern in the randomized order
        for( q = 0 ; q < train_data_num ; q++ ) 
        {    
            p = RandomizedIndex[q];

            // Compute hidden layer activations
            for( i = 0 ; i < HiddenNodes ; i++ ) {    
                Accum = HiddenWeights[input_length][i] ;
                for( j = 0 ; j < input_length ; j++ ) {
                    Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
                }
                Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
            }

            // Compute output layer activations and calculate errors
            for( i = 0 ; i < target_num ; i++ ) {    
                Accum = OutputWeights[HiddenNodes][i] ;
                for( j = 0 ; j < HiddenNodes ; j++ ) {
                    Accum += Hidden[j] * OutputWeights[j][i] ;
                }
                Output[i] = 1.0/(1.0 + exp(-Accum)) ;   
                OutputDelta[i] = (train_data_output[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;   
                Error += 0.5 * (train_data_output[p][i] - Output[i]) * (train_data_output[p][i] - Output[i]) ;
            }

            // Backpropagate errors to hidden layer
            for( i = 0 ; i < HiddenNodes ; i++ ) {    
                Accum = 0.0 ;
                for( j = 0 ; j < target_num ; j++ ) {
                    Accum += OutputWeights[i][j] * OutputDelta[j] ;
                }
                HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
            }

            // Update Input-->Hidden Weights
            for( i = 0 ; i < HiddenNodes ; i++ ) {     
                ChangeHiddenWeights[input_length][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[input_length][i] ;
                HiddenWeights[input_length][i] += ChangeHiddenWeights[input_length][i] ;
                for( j = 0 ; j < input_length ; j++ ) { 
                    ChangeHiddenWeights[j][i] = LearningRate * train_data_input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
                    HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
										//HiddenWeights[j][i] = sqrt(6.0 / (input_length + HiddenNodes)) * (2.0 * Rando - 1.0);
                }
            }

            // Update Hidden-->Output Weights
            for( i = 0 ; i < target_num ; i ++ ) {    
                ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
                OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
                for( j = 0 ; j < HiddenNodes ; j++ ) {
                    ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
                    OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
										//OutputWeights[j][i] = sqrt(6.0 / (HiddenNodes + target_num)) * (2.0 * Rando - 1.0);
                }
            }
        }
        accuracy = Get_Train_Accuracy();

        // Every 10 cycles send data to terminal for display
        ReportEvery10 = ReportEvery10 - 1;
        if (ReportEvery10 == 0)
        {
            
            printf ("\nTrainingCycle: %ld\n",TrainingCycle);
            printf ("Error = %.5f\n",Error);
            printf ("Accuracy = %.2f /100 \n",accuracy*100);
            //run_train_data();

            if (TrainingCycle==1)
            {
                ReportEvery10 = 9;
            }
            else
            {
                ReportEvery10 = 10;
            }
        }

        // If error rate is less than pre-determined threshold then end
        if( accuracy >= goal_acc ) break ;
    }

    printf ("\nTrainingCycle: %ld\n",TrainingCycle);
    printf ("Error = %.5f\n",Error);
    run_train_data();
    printf ("Training Set Solved!\n");
    printf ("--------\n"); 
    printf ("Testing Start!\n ");
    run_test_data();
    printf ("--------\n"); 
    ReportEvery10 = 1;
    load_weight();
		
		printf("\nModel Training Phase has ended.\n");

    /* Start prediction */
    //AdcSingleCycleScanModeTest();

    //while(1);
}