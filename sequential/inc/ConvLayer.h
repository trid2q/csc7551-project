#ifndef _CONV_LAYER_
#define _CONV_LAYER_

#include "Filter.h"
#include "Layer.h"
#include "Gradient.h"
#include "Pixel.h"

namespace edu {
namespace tri {
namespace cnn {

class ConvLayer: public Layer {
    public:
        ConvLayer(int imageW, int imageD, int nFilters, int filterW) {
    	    assert(filterW%2 == 1); //  This code work only with odd filter width 
            deltas_ = new Volume<float>(imageW, imageD);
            
	    for (int i = 0; i < nFilters; ++i) {
		Filter* filter = new Filter(filterW, imageD);
		filter->randomize();  // Populate filter weights
		filters_.push_back(filter);
                Volume<Gradient>* gradient = new Volume<Gradient>(filterW, imageD);
                filtersGradients_.push_back(gradient);
	    }
	    filterCenter_.i(filterW/2);
	    filterCenter_.j(filterW/2);
	    out_ = new Volume<float>(imageW, nFilters);
        }
        virtual ~ConvLayer() {
	    for (int i = 0; i < filtersGradients_.size(); ++i) {
		delete filtersGradients_[i];
		delete filters_[i];
	    }
        }
        virtual Volume<float>* forward(Volume<float>* in);
	virtual Volume<float>* computeError(Volume<float>* nextLayerWeights, Volume<float>* nextLayerDeltas); 
        virtual void adjustWeights();
        virtual void load(ifstream& ifile);
        virtual void save(ofstream& ofile);
        

    private:
        ConvLayer() {}
        bool isOverLap(int i, int j, int m, int n);

    private:
        vector<Filter*> filters_;
        vector<Volume<Gradient>*> filtersGradients_;
	Pixel filterCenter_;
};

}
}
}

using namespace edu::tri::cnn;

#endif
