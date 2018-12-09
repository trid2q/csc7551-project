#ifndef _POOL_LAYER_
#define _POOL_LAYER_

#include "Layer.h"

namespace edu {
namespace tri {
namespace cnn {

class PoolLayer: public Layer {
    public:
        PoolLayer(int nWinSize, Volume<float>* prevOut): nWinSize_(nWinSize) {
	    out_ = new Volume<float>(prevOut->width()/nWinSize, prevOut->depth());
            deltas_ = new Volume<float>(prevOut->width(), prevOut->height(), prevOut->depth());
        }
        virtual ~PoolLayer() {
        }
        virtual Volume<float>* forward(Volume<float>* in);
        virtual Volume<float>* computeError(Volume<float>* forwardWeights, Volume<float>* forwardDeltas); 
        virtual void adjustWeights() {};

    private:
        int nWinSize_;
};

}
}
}

using namespace edu::tri::cnn;

#endif
