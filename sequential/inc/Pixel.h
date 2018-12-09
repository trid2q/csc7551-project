#ifndef _PIXEL_
#define _PIXEL_

namespace edu {
namespace tri {
namespace cnn {

class Pixel {
    public:
        Pixel() {
            i_ = 0;
            j_ = 0;
        }
    Pixel(int i, int j): i_(i), j_(j) {
    }
    Pixel(Pixel& src): i_(src.i_), j_(src.j_) {
    }
    virtual ~Pixel() {
    }
    int i() {
       return i_;
    }
    void i(int nI) {
       i_ = nI;
    }
    int j() {
       return j_;
    }
    void j(int nJ) {
       j_ = nJ;
    }
    Pixel& operator=(const Pixel& src) {
        i_ = src.i_;
        j_ = src.j_;
        return *this;
    }
    private:    
        int i_;
        int j_;
};

}
}
}

#endif

