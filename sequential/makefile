CC=g++
RM=rm
CFLAGS=-std=gnu++0x
TARGET=cnn

INCLUDES = -I./ -I./inc

default: all

SRCS = main.cpp Cnn.cpp Matrix.cpp Volume.cpp ConvLayer.cpp ReLuLayer.cpp PoolLayer.cpp FCLayer.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET) : $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

#.cpp.o:
%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	$(RM) -rf *.o
