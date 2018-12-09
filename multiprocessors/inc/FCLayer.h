#ifndef _FC_LAYER_
#define _FC_LAYER_

#include <fstream>
#include "Layer.h"
#include "Gradient.h"

namespace edu {
namespace tri {
namespace cnn {

class FCLayer : public Layer {
    public:
        FCLayer(int nNodes, Volume<float>* prevOut) {
            assert(nNodes > 0);
            nNodes_ = nNodes;
            nodeSumIn_.add(new Matrix<float>(nNodes_, 1));
    	    nodeSumIn_.clear();
	    gradients_ = new Gradient[nNodes];
	weights_ = new Volume<float>(prevOut->size(), nNodes_, 1);
        int maxval = prevOut->size();
        for (int d = 0; d < weights_->depth(); ++d) {
            for (int i = 0; i < weights_->height(); ++i) {
                for (int j = 0; j < weights_->width(); ++j) {
                    //weights_->setValue(i, j, d, T(rand()%10)); //1.0f / maxval * rand() / float( RAND_MAX );
                    (*weights_)(i, j, d) = 2.19722f / maxval * rand() / float( RAND_MAX );
                }
            }
	}
    out_ = new Volume<float>(nNodes_, 1, 1);
    deltas_ = new Volume<float>(prevOut->width(), prevOut->height(), prevOut->depth());
        }
        virtual ~FCLayer() {
        }
	virtual Volume<float>* forward(Volume<float>* in);
        virtual Volume<float>* computeError(Volume<float>* forwardWeights, Volume<float>* forwardDeltas);
        virtual void adjustWeights();
        virtual void save(ofstream& afile) {
	    Layer::save(afile);
	    for (int i = 0; i < nNodes_; ++i) {
		afile << gradients_[i].old() << gradients_[i].value();
            }
	    nodeSumIn_.save(afile);
	}
        virtual void load(ifstream& afile) {
	    Layer::load(afile);
	    for (int i = 0; i < nNodes_; ++i) {
		float oldGrad, newGrad;
		afile >> oldGrad >> newGrad;
		gradients_[i].old(oldGrad);
		gradients_[i].value(newGrad);
            }
	    nodeSumIn_.load(afile);
	}

    private:
        int nNodes_;
        Gradient* gradients_;
        Volume<float> nodeSumIn_;
};

}
}
}

#endif
