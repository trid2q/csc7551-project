#include "ConvLayer.h"

#include "Volume.h"

namespace edu {
namespace tri {
namespace cnn {

Volume<float>* ConvLayer::computeError(Volume<float>* nextLayerWeights, Volume<float>* nextLayerDeltas) {
    // For each filter
    for (int k = 0; k < filters_.size(); ++k) {
	Filter& filter = *filters_[k];
        Volume<Gradient>* filterGrads = filtersGradients_[k];
	// For each image row
	for (int i = 0; i < in_->height(); ++i) {
        // For each pixel in image row
            for (int j = 0; j < in_->width(); ++j) {
                for (int d = 0; d < in_->depth(); ++d) {
	          float errorSum = 0.0f;
		  // For each kernel row in kernel
		  for (int m = 0; m < filter.height(); ++m)  {
		    // For each elment in kernel row
		    for (int n = 0; n < filter.width(); ++n) {
                        if (isOverLap(i, j, m, n)) {
		                errorSum += filter(m, n, d)*(*nextLayerDeltas)(i,j,k);
				Gradient gradient = (*filterGrads)(m, n, d);
                                gradient.value((*in_)(i,j,d)*(*nextLayerDeltas)(i,j,k));
                                (*filterGrads)(m, n, d) = gradient;
			}
		    }
		    (*deltas_)(i, j, d) =  errorSum;
                  }
		}
	    }
	}
    }
    return deltas_;
}

void ConvLayer::adjustWeights() {
    for (int k = 0; k < filters_.size(); ++k) {
        Filter& filter = *(filters_[k]);
	Volume<Gradient>& filterGrads = *(filtersGradients_[k]);
	for (int i = 0; i < filter.height(); ++i) {
	    for (int j = 0; j < filter.width(); ++j) {
		for (int d = 0; d < filter.depth(); ++d) {
		    float w = filter(i, j, d);
		    Gradient gradient = filterGrads(i, j, d);
		    float newWeight = gradient.adjustWeight(w);
		    filter(i, j, d) = newWeight;
                    gradient.update();
		}
	    }
	}
    }
}
		    
bool ConvLayer::isOverLap(int i, int j, int m, int n) {
    bool bOverLap = false;
    int filterI = -1;
    int filterJ = -1;
    if (m != filterCenter_.i()) {
	filterI = i + (m - filterCenter_.i());
    } else {
	filterI = i;
    }
    if (n != filterCenter_.j()) {
	filterJ = j + (n - filterCenter_.j());
    } else {
	filterJ = j;
    }
    if (filterI >= 0 && filterI < in_->height()) {
	if (filterJ >= 0 && filterJ < in_->width()) {
	   bOverLap = true;
	}
    }
    return bOverLap;
}
/*
for each image row in input image:
   for each pixel in image row:

      set accumulator to zero

      for each kernel row in kernel:
         for each element in kernel row:

            if element position  corresponding* to pixel position then
               multiply element value  corresponding* to pixel value
               add result to accumulator
            endif

      set output image pixel to accumulator
*/
Volume<float>* ConvLayer::forward(Volume<float>* in) {
    in_ = in;
    Filter* filter = filters_.front();
    assert(filter->width() == filter->height());
    assert(filter->width()%2 == 1);
    int filterCenterM = filter->height()/2;
    int filterCenterN = filter->width()/2;
    // For each filter
    for (int k = 0; k < filters_.size(); ++k) {
	Matrix<float>& featureMap = (*out_)[k];
	filter = filters_[k];
	// For each image row
	for (int i = 0; i < in_->height(); ++i) {
        // For each pixel in image row
            for (int j = 0; j < in_->width(); ++j) {
	        float sum = 0.0f;
		// For each kernel row in kernel
		for (int m = 0; m < filter->height(); ++m)  {
		    // For each elment in kernel row
		    for (int n = 0; n < filter->width(); ++n) {
                        if (isOverLap(i, j, m, n)) {
		            for (int d = 0; d < in_->depth(); ++d) {
			       sum += (*filter)(m, n, d)*(*in_)(i + m - filterCenterN, j+n-filterCenterN, d);
			    }
			}
		    }
		}
                featureMap(i, j) = sum;
	    }
	}
    }
    return out_;
}

void ConvLayer::save(ofstream& ofile) {
    Layer::save(ofile);
    int nLayer = 0;
    for (int i = 0; i < filtersGradients_.size(); ++i) {
        filtersGradients_[i]->save(ofile);
   }
}
void ConvLayer::load(ifstream& ifile) {
    int nLayer = 0;
    Layer::load(ifile);
    for (int i = 0; i < filtersGradients_.size(); ++i) {
        filtersGradients_[i]->load(ifile);
   }
}


}
}
}


