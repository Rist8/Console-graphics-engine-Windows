# Console-graphics-engine-Windows
Console graphics engine writen using &lt;Windows.h> for smooth output (couldn't find any alternative for Linux)
f11 to open console in full screen mode
to build just compile Engine.cpp using MSVC compiler in any way you want

# Updated
Added version for linux with Cmake file including 2 options: `single` and `multi`
to build it clone repository and from it follow the instructions:
```
mkdir build && cd build
cmake ..
make single    # Build without OpenMP
make multi     # Build with OpenMP (if available)
```
