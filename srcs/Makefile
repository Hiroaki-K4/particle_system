PARENT_DIR := /home/h-kubo/mypro/
SRCS := main.cpp Particle.cu Shader.cpp glad.c kernel.cu
INCLUDE := -I../glfw/include -I../glad/include -I../glm
LDFLAGS := -L$(PARENT_DIR)particle_system/glfw/build/src `pkg-config --libs glfw3` -lglfw3 -lGL -lX11 -lpthread -lXrandr -lXi -ldl
NAME := particle_system
CXX := nvcc

all: $(NAME)

$(NAME): $(SRCS)
	$(CXX) $(SRCS) $(INCLUDE) $(LDFLAGS) -o $(NAME)

clean:
	rm -rf $(NAME)

re: clean all

.PHONY: all clean re
