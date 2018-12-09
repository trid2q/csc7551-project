#include "PoolLayer.h"

#include "Volume.h"

namespace edu {
namespace tri {
namespace cnn {

Volume<float>* PoolLayer::computeError(Volume<float>* nextLayerWeights, Volume<float>* nextLayerDeltas) {
    for (int d = 0; d < in_->depth(); ++d) {
        Matrix<float>* mDelta = deltas_->get(d);
	Matrix<float> mForwardDelta = *(nextLayerDeltas->get(d));
	Matrix<float>& mIn = *(in_->get(d));
        Matrix<float>& mOut = *(out_->get(d));

	int i, j;
        #pragma omp parallel private(i, j)
        #pragma omp for schedule(dynamic)
	for (i = 0; i < mIn.height(); ++i) {
           int outI = i/nWinSize_;
	   for (int j = 0; j < mIn.width(); ++j) {
               int outJ = j/nWinSize_;
               if (mIn(i,j) == mOut(outI, outJ)) {
                   (*mDelta)(i, j) = mForwardDelta(outI, outJ);
	       } else {
		   (*mDelta)(i, j) = 0;
	       }
	   }
	}
    }
    return deltas_;
}

Volume<float>* PoolLayer::forward(Volume<float>* in) {
    in_ = in;
    assert(in_->width() % nWinSize_ == 0);
    assert(in_->height() % nWinSize_ == 0);
    int d, i, j;
    #pragma omp parallel private(d, i, j)
    for (d = 0; d < in_->depth(); ++d) {
        #pragma omp for schedule(dynamic)
	for (i = 0; i < in_->height(); i += nWinSize_) {
	   for (j = 0; j < in_->width(); j += nWinSize_) {
               float fMax = maxPooling(i, j, d);
	       out_->setValue(i/nWinSize_, j/nWinSize_, d, fMax);
	   }
	}
    }
    return out_;
}

float PoolLayer::maxPooling(int leftCornerI, int leftCornerJ, int depth) {
    assert(leftCornerI >= 0);
    assert(leftCornerI + nWinSize_ <= in_->height());
    assert(leftCornerJ + nWinSize_ <= in_->width());
    assert(depth >= 0 && depth < in_->depth());
    float fMax = numeric_limits<float>::min();
    for (int i = leftCornerI; i < leftCornerI + nWinSize_; ++i) {
        for (int j = leftCornerJ; j < leftCornerJ + nWinSize_; ++j) {
	     fMax = ::max(fMax, in_->value(i, j, depth));
	}
    }
    return fMax;
}

}
}
}


