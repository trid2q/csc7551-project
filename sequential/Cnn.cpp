#include "Cnn.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include "ConvLayer.h"
#include "ReLuLayer.h"
#include "PoolLayer.h"
#include "FCLayer.h"

namespace edu {
namespace tri {
namespace cnn {

float Cnn::computeMeanSquareError(Volume<float>* errors) {
    float sumErr = 0.0f;
    for (int d = 0; d < errors->depth(); ++d) {
	Matrix<float>* error = errors->get(d);
	for (int i = 0; i < error->height(); ++i) {
	    for (int j = 0; j < error->width(); ++j) {
                float err = error->value(i,j);
                sumErr += pow(err,2);
	    }
	}
    }
    return (sumErr/errors->size());
}

float Cnn::train(Volume<float>* image, Volume<float>* expectedOut) {
    bool bDone = false;
    Volume<float>* in = image;
    Volume<float>* out = in;
    // Forward propagation
    for (vector<Layer*>::iterator it = layers_.begin(); it != layers_.end(); ++it) {
    	out = (*it)->forward(out);
    }
    *deltas_ = *out;
    *deltas_ -= *expectedOut;

    Volume<float>* forwardDeltas = deltas_;
    float fError = computeMeanSquareError(forwardDeltas);
    // Backward Propagation
    Volume<float>* forwardWeights = layers_.back()->weights();
    for (vector<Layer*>::reverse_iterator rit = layers_.rbegin(); rit != layers_.rend(); ++rit) {
        forwardDeltas = (*rit)->computeError(forwardWeights, forwardDeltas);
    }	
    for (vector<Layer*>::iterator it = layers_.begin(); it != layers_.end(); ++it) {
        (*it)->adjustWeights();
    }	
    return fError;
}

void Cnn::forward(Volume<float>& image) {
    Volume<float>* out = &image;
    for (vector<Layer*>::iterator it = layers_.begin(); it != layers_.end(); ++it) {
    	out = (*it)->forward(out);
    }
}

void Cnn::save(string fname) {
    ofstream ofile(fname);
    for (vector<Layer*>::iterator it = layers_.begin(); it != layers_.end(); ++it) {
    	(*it)->save(ofile);
    }
    layers_.back()->saveDeltas(ofile);
}
void Cnn::load(string fname) {
    ifstream ifile(fname);
    for (vector<Layer*>::iterator it = layers_.begin(); it != layers_.end(); ++it) {
    	(*it)->load(ifile);
    }
    layers_.back()->loadDeltas(ifile);
}

}
}
}

