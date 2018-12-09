#include "ConvLayer.h"

#include "Volume.h"

namespace edu {
namespace tri {
namespace cnn {

Volume<float>* ConvLayer::computeError(Volume<float>* nextLayerWeights, Volume<float>* nextLayerDeltas) {
    int i, j, F;
    //Create an array of all input pixel locations
    int nNumPixels = in_->width()*in_->height();
    Pixel* pixels = new Pixel[nNumPixels];
    #pragma omp parallel shared(pixels, nNumPixels) private(i, j, F)
    #pragma omp for schedule(dynamic)
    for (i = 0; i < nNumPixels; ++i) {
        int mI = i / in_->width();
        int mJ = i % in_->width();
        pixels[i] = Pixel(mI, mJ);
    }

    for (int k = 0; k < filters_.size(); ++k) {
	Filter& filter = *filters_[k];
	Volume<Gradient>& filterGrads = *(filtersGradients_[k]);
        for (int d = 0; d < in_->depth(); ++d) {
          #pragma omp for schedule(dynamic)
    	  for (int p = 0; p < nNumPixels; ++p) {
            float errorSum = convolveErrors(pixels[p], filter[d],
		                    (*in_)[d], filterGrads[d], (*nextLayerDeltas)[k]);
            (*deltas_)(pixels[p].i(), pixels[p].j(),d) = errorSum;
    	  }
        }
    }
    
    delete []pixels;
    return deltas_;
}

float ConvLayer::convolveErrors(Pixel& pixel, Matrix<float>& mFilter, Matrix<float>& mIn,
	Matrix<Gradient>& mFilterGrad, Matrix<float>& mNextLayerDelta) {
    float sum = 0.0f;
    // For each kernel row in kernel
    for (int m = 0; m < mFilter.height(); ++m)  {
	// For each elment in kernel row
        for (int n = 0; n < mFilter.width(); ++n) {
            if (isOverLap(pixel.i(), pixel.j(), m, n)) {
 	        sum += mFilter.value(m, n)*mNextLayerDelta.value(pixel.i(), pixel.j());
                Gradient gradient = mFilterGrad(m, n);
                gradient.value(mIn(pixel.i(), pixel.j())*mNextLayerDelta(pixel.i(), pixel.j()));
                mFilterGrad(m, n) = gradient;
	    }
	}
    }
    return sum;
}
float ConvLayer::convolve(Pixel& pixel, Filter& filter) {
    int filterCenterM = filter.height()/2;
    int filterCenterN = filter.width()/2;
    float sum = 0.0f;
    // For each kernel row in kernel
    for (int m = 0; m < filter.height(); ++m)  {
	// For each elment in kernel row
        for (int n = 0; n < filter.width(); ++n) {
            if (isOverLap(pixel.i(), pixel.j(), m, n)) {
                for (int d = 0; d < in_->depth(); ++d) {
		    float elmtwizeProduct = filter.value(m, n, d)*
				            in_->value(pixel.i() + m - filterCenterN, pixel.j()+n-filterCenterN, d);
 	            sum += elmtwizeProduct;
	        }
	    }
	}
    }
    return sum;
}
void ConvLayer::adjustWeights() {
    Filter& filter = *(filters_.front());
    int nFilterMatrixSize = filter.width()*filter.height();
    int nNumWeights = filter.size();
    for (int k = 0; k < filters_.size(); ++k) {
        filter = *filters_[k];
	Volume<Gradient>& filterGrads = *(filtersGradients_[k]);
	int i;
	#pragma omp paralell for private(i) schedule(dynamic)
        for (i = 0; i < nNumWeights; ++i) {
            int mD = i / nFilterMatrixSize;
            int mI = (i % nFilterMatrixSize) / filter.width();
	    int mJ = (i % nFilterMatrixSize) % filter.width(); 
	    float w = filter(mI, mJ, mD);
            Gradient gradient = filterGrads(mI, mJ, mD);
	    float newWeight = gradient.adjustWeight(w);
	    filter(mI, mJ, mD) = newWeight;
            gradient.update();
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
    int i, j;
    in_ = in;
    //Create an array of all input pixel locations
    int nNumPixels = in_->width()*in_->height();
    Pixel* pixels = new Pixel[nNumPixels];
    #pragma omp parallel shared(pixels, nNumPixels) private(i, j)
    #pragma omp for schedule(dynamic)
    for (i = 0; i < nNumPixels; ++i) {
        int mI = i / in_->width();
        int mJ = i % in_->width();
        pixels[i] = Pixel(mI, mJ);
    }
//    Volume* featureMaps = new Volume();
    for (int k = 0; k < filters_.size(); ++k) {
	Filter& filter = *filters_[k];
        Matrix<float>& featureMap = (*out_)[k];
    	// Apply convolution
        #pragma omp for schedule(dynamic)
    	for (int p = 0; p < nNumPixels; ++p) {
	    featureMap(pixels[p].i(), pixels[p].j()) = convolve(pixels[p],filter);
    	}
    }
    
    delete []pixels;
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


