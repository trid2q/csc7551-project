#ifndef _RE_LU_LAYER_
#define _RE_LU_LAYER_

#include "Layer.h"
#include "Volume.h"

namespace edu {
namespace tri {
namespace cnn {

class ReLuLayer: public Layer {
    public:
        ReLuLayer(Volume<float>* prevOut) {
	   assert(prevOut);
	   out_ = new Volume<float>(prevOut->width(), prevOut->depth());
           deltas_ = new Volume<float>(prevOut->width(), prevOut->depth());
        }
        virtual ~ReLuLayer() {
        }
        virtual Volume<float>* forward(Volume<float>* in);
	virtual Volume<float>* computeError(Volume<float>* forwardWeights, Volume<float>* forwardDeltas); 
        virtual void adjustWeights() {}
};

}
}
}

using namespace edu::tri::cnn;

#endif
