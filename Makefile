PARENT_DIR := /home/h-kubo/mypro/
SRCS := main.cpp glad.c
INCLUDE := -Iglfw/include -Iglad/include
LDFLAGS := -L$(PARENT_DIR)particle_system/glfw/build/src `pkg-config --libs glfw3` -lglfw3 -lGL -lX11 -lpthread -lXrandr -lXi -ldl
NAME := particle_system
CXX := g++

all: $(NAME)

$(NAME): $(SRCS)
	$(CXX) $(SRCS) $(INCLUDE) $(LDFLAGS) -o $(NAME)

clean:
	rm -rf $(NAME)

re: clean all

.PHONY: all clean re
