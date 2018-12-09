#include <omp.h>

#include <cstdint>
#include <fstream>
#include "Cnn.h"
#include "ConvLayer.h"
#include "ReLuLayer.h"
#include "PoolLayer.h"
#include "FCLayer.h"
#include "Case.h"

uint32_t byteswap_uint32(uint32_t a) {
        return ((((a >> 24) & 0xff) << 0) |
                (((a >> 16) & 0xff) << 8) |
                (((a >> 8) & 0xff) << 16) |
                (((a >> 0) & 0xff) << 24));
}

uint8_t* read_file( const char* szFile ) {
        ifstream file( szFile, ios::binary | ios::ate );
        streamsize size = file.tellg();
        file.seekg( 0, ios::beg );
        if ( size == -1 ) return NULL;
        uint8_t* buffer = new uint8_t[size];
        file.read( (char*)buffer, size );
        return buffer;
}

void readTestCases(vector<Case*>& testCases) {
        uint8_t* train_image = read_file( "../data/train-images.idx3-ubyte" );
        uint8_t* train_labels = read_file( "../data/train-labels.idx1-ubyte" );
        uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );
	// For every case, read its image and expected output
	int imageW = 28;
	int filterW = 3;
	int nClasses = 10;
        for ( int nCase = 0; nCase < case_count; ++nCase ) {
                uint8_t* img = train_image + 16 + nCase * (imageW*imageW);
                uint8_t* label = train_labels + 8 + nCase;
		// Read an image
                Volume<float>* vImage = new Volume<float>();
		Matrix<float>* mImage = new Matrix<float>(imageW, imageW);
		for (int i = 0; i < imageW; ++i) {
		    for (int j = 0; j < imageW; ++j) {
			(*mImage)(i, j) =  img[i+j*imageW]/255.0f;
		    }
		}
		vImage->add(mImage);
		// Read expected output
		Volume<float>* vOutput = new Volume<float>();
		Matrix<float>* mOutput = new Matrix<float>(nClasses,1);
		for (int nOut = 0; nOut < nClasses; ++nOut) {
		    if (*label == nOut) {
		        (*mOutput)(0,nOut) = 1.0f;
		    } else {
			(*mOutput)(0,nOut) = 0.0f;
		    }
		} 
		vOutput->add(mOutput);
		testCases.push_back(new Case(vImage, vOutput));
        }
        delete[] train_image;
        delete[] train_labels;
        cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ": # Cases:" << testCases.size() << endl;
}

void setup(Cnn& cnn, vector<Case*>& testCases) { 
    // Setup Cnn
    int nFilters = 8;
    int nFilterW = 3;
    int imageW = testCases[0]->image()->width();
    int imageD = 1;

    ConvLayer* layer1 = new ConvLayer(imageW, imageD, nFilters, nFilterW); 
    cnn.append(layer1);
    ReLuLayer* layer2 = new ReLuLayer(layer1->output());
    cnn.append(layer2);

    int nPoolingWindowW = 2;
    PoolLayer* layer3 = new PoolLayer(nPoolingWindowW, layer2->output()); 
    cnn.append(layer3);

    int nClasses = 10;
    FCLayer* layer4 = new FCLayer(nClasses, layer3->output());
    cnn.append(layer4);
}

void train(Cnn& cnn, vector<Case*>& testCases, int nEpochs) {
    float fError = 1.0f;
    int epoch = 0;
    int nCase = 0;
    float accumulateMSE = 0.0f; 
    int nTrains = 0;
    bool bComplete = false;
    for (epoch = 0; epoch < nEpochs; ++epoch) {
	for (nCase = 0; nCase < testCases.size(); ++nCase) {
	    fError = cnn.train(testCases[nCase]->image(), testCases[nCase]->output());
	    accumulateMSE += fError;
	    ++nTrains;
            if ((nCase+1)%200 == 0) {
                cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ": MSE from last image: " << fError
                     <<", Accu MSE = " << accumulateMSE  
	             << ", Epoch #" << epoch + 1 << ", #cases = " << nTrains << ", Ave Accu MSE = "
                     << accumulateMSE/nTrains << endl;
            }
	}
    }
}

bool test(Cnn& cnn) {
    bool bCorrect = false;
        //while ( true )
        {
                uint8_t * data = read_file( "../data/test.ppm" );

                if ( data )
                {
                        uint8_t * usable = data;

                        while ( *(uint32_t*)usable != 0x0A353532 )
                                usable++;

#pragma pack(push, 1)
                        struct RGB
                        {
                                uint8_t r, g, b;
                        };
#pragma pack(pop)

                        RGB * rgb = (RGB*)usable;

                        Volume<float> image(28, 28, 1);
                        for ( int i = 0; i < 28; i++ )
                        {
                                for ( int j = 0; j < 28; j++ )
                                {
                                        RGB rgb_ij = rgb[i * 28 + j];
                                        image( j, i, 0 ) = (((float)rgb_ij.r
                                                             + rgb_ij.g
                                                             + rgb_ij.b)
                                                            / (3.0f*255.f));
                                }
                        }
			
			cnn.forward(image);
			Volume<float>& actualOut = cnn.output();
                        cout << __FILE__ << ":" << __func__ << ":" << __LINE__ 
                             << ":===== actual out:" << actualOut << endl;
                        float fMaxActualOut = 0.0f;
			int nClass = -1;
                        for ( int i = 0; i < 10; i++ )
                        {
                                float fActualOut = actualOut( 0, i, 0 )*100.0f;
				if (fActualOut > fMaxActualOut) {
				    fMaxActualOut = fActualOut;
				    nClass = i;
				}
                                printf( "[%i] %f\n", i, fActualOut); //actualOut( 0, i, 0 )*100.0f );
                        }
                        delete[] data;
			if (nClass == 8) {  // Wecause of time conssumtion, we test with rumber 8 only
			    bCorrect = true;
                            cout << "========= YES !!!! It's #8.   COMPLETE ======" << endl;
			} else {
                            cout << "========= OH NO!!!! It's #" << nClass << ".  CONTINUE TO TRAIN...======" << endl;
			}
                }
        }
	return bCorrect;
}

void printUsage() {
    cout << "Usage: ./cnn -c <train/test> -cnn <cnnAfterTrainDataFile>" << endl;
    cout << "COMMENT: train option does both train and test" << endl;
    cout << "         test option does test only but not working yet" << endl;
    cout << "         For now, just do:  \"cnn\" to run" << endl;
    cout << "Ex: ./cnn -c train -cnn saveToFileName" << endl;
    cout << "    ./cnn -c test -cnn fileToLoadFrom" << endl;
    cout << "    ./cnn" << endl;
}

int main(int argc, char* argv[]) {
   srand(1);
   string cmd = "all";
   string cnnFile = "";
   if (argc > 1) {
       if (argc <5) {
           printUsage();
           return 0;
       }
   
       for (int i = 1; i < argc; ++i) {
           string arg = argv[i];
	   if (arg == "-c") {
	       cmd = argv[++i];
	   } else if (arg == "-cnn") {
	       cnnFile = argv[++i];
	   } 
       }
       if (cnnFile == "") {
	   printUsage();
           return 0;
       }
   }
   //Read test cases
   vector<Case*> testCases;
   readTestCases(testCases);

   Cnn cnn;
   setup(cnn, testCases);
   omp_set_num_threads(28);
   if (cmd == "all" || cmd == "train") {
	bool bRecognize = false;
	double trainedTime = 0;
	int nEpochs = 0;
        cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ":===== PARAMETERS: " << endl;
        cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ":     Image Volume      :  28 x 28 x 1" << endl;
        cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ":     Filter Volume     :  3 x 3 x 1" << endl;
        cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ":     # Filters         :  8" << endl;
        cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ":     Zero padding      :  1" << endl;
        cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ":     Striding          :  1" << endl;
        cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ":     Pooling window    :  2 x 2" << endl;
        cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ":     Reclinear function: max(0,x)" << endl;

        cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ":===== IN TRAINING..." << endl;
        int count = 0;
	do {
	    int nEpoch = 1;
            cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << "++++++++++++++ epoch " << nEpochs << endl;
            double startTime = omp_get_wtime();
            train(cnn, testCases, nEpoch);
            trainedTime += omp_get_wtime() - startTime;
	    nEpochs += nEpoch;
	    bRecognize = test(cnn);
	    cout << ">>>>>> Total training tme = " << trainedTime << " secs, #Epochs = " << nEpochs
                 << ", Recognized = " << bRecognize << endl;
	} while (++count < 3); //!bComplete);

        // Free test cases
        for (int nCase = 0; nCase < testCases.size(); ++nCase) {
	    delete testCases[nCase];
        }
        if (cnnFile != "") {
            cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ": Save CNN to: " << cnnFile << endl;
            cnn.save(cnnFile); // + ".new");
        }
    } if (cmd == "test") {
        cout << __FILE__ << ":" << __func__ << ":" << __LINE__ << ": Load CNN from: " << cnnFile << endl;
	cnn.load(cnnFile);
	test(cnn);
    } 
   
   return 0;
}
