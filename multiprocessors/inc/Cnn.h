#ifndef _CNN_
#define _CNN_

#include "Layer.h"
#include "Volume.h"

namespace edu {
namespace tri {
namespace cnn {

class Cnn {
    public:
	static const float ERROR_THRESHOLD = 0.02f;
        Cnn() {
	    deltas_ = new Volume<float>(10, 1, 1);
        }
        virtual ~Cnn() {
	    for (vector<Layer*>::iterator it = layers_.begin(); it != layers_.end(); ++it) {
	        delete *it;
	    }
        }
        float train(Volume<float>* image, Volume<float>* expectedOut);
	void forward(Volume<float>& image);
	Volume<float>& output() {
	    return *(layers_.back()->output());
	}
        void append(Layer* layer) {
 	    layers_.push_back(layer);
	}
	void save(string fname);
	void load(string fname);

    private:
	float computeMeanSquareError(Volume<float>* errors);

    private:
	vector<Layer*> layers_;
        Volume<float>* deltas_;
};

}
}
}

using namespace edu::tri::cnn;

#endif
