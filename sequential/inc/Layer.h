#ifndef _LAYER_
#define _LAYER_

#include "Filter.h"
#include "Volume.h"

namespace edu {
namespace tri {
namespace cnn {

class Layer;

class Layer {
    public:
        Layer() {
            in_ = NULL;
            out_ = NULL;
            weights_ = NULL;
	    deltas_ = NULL;
        }
        virtual ~Layer() {
           delete weights_;
	   delete deltas_;
	   delete out_;
	   in_ = NULL; //out_ = NULL;
        }
        virtual Volume<float>* forward(Volume<float>* in) = 0;
        virtual Volume<float>* computeError(Volume<float>* nextLayerWeights, Volume<float>* nextLayerDeltas) = 0;
        virtual void adjustWeights() = 0;

	Volume<float>* weights() {
	    return weights_;
	}
	Volume<float>* deltas() {
	    return deltas_;
	}
        Volume<float>* output() const {
	    return out_;
	}
        virtual void save(ofstream& ofile) {
	    if (weights_) {
	        weights_->save(ofile);
	    }
	    //deltas_->save(ofile);
        }	
	virtual void load(ifstream& ifile) {
    cout << __FILE__ << ":" << __func__ << ":" << __LINE__  << endl;
	    if (weights_) {
    cout << __FILE__ << ":" << __func__ << ":" << __LINE__  << endl;
	        weights_->load(ifile);
    cout << __FILE__ << ":" << __func__ << ":" << __LINE__  << endl;
            }
    cout << __FILE__ << ":" << __func__ << ":" << __LINE__  << endl;
	    //deltas_->load(ifile);
    cout << __FILE__ << ":" << __func__ << ":" << __LINE__  << endl;
	}
	void saveDeltas(ofstream& ofile) {
	    deltas_->save(ofile);
	}
	void loadDeltas(ifstream& ifile) {
	    deltas_->load(ifile);
	}
	    

    protected:
        Volume<float>* in_;
        Volume<float>* out_;
        Volume<float>* weights_;
        Volume<float>* deltas_;
};

}
}
}

#endif
