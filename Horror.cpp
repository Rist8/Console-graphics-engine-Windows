#include <iostream>
#include <stdio.h>
#include <math.h>
#include <functional>
#include <conio.h>
#include <windows.h>



struct vec2
{
	float x, y;

	vec2(float value) : x(value), y(value) {}
	vec2(float _x, float _y) : x(_x), y(_y) {}

	vec2 operator+(vec2 const& other) { return vec2(x + other.x, y + other.y); }
	vec2 operator-(vec2 const& other) { return vec2(x - other.x, y - other.y); }
	vec2 operator*(vec2 const& other) { return vec2(x * other.x, y * other.y); }
	vec2 operator/(vec2 const& other) { return vec2(x / other.x, y / other.y); }
};

struct vec3
{
	float x, y, z;

	vec3(float _value) : x(_value), y(_value), z(_value) {};
	vec3(float _x, vec2 const& v) : x(_x), y(v.x), z(v.y) {};
	vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {};

	vec3 operator+(vec3 const& other) { return vec3(x + other.x, y + other.y, z + other.z); }
	vec3 operator-(vec3 const& other) { return vec3(x - other.x, y - other.y, z - other.z); }
	vec3 operator*(vec3 const& other) { return vec3(x * other.x, y * other.y, z * other.z); }
	vec3 operator/(vec3 const& other) { return vec3(x / other.x, y / other.y, z / other.z); }
	vec3 operator-() { return vec3(-x, -y, -z); }

};

float clamp(float value, float min, float max) { return fmax(fmin(value, max), min); }
double sign(double a) { return (0 < a) - (a < 0); }
double step(double edge, double x) { return x > edge; }

float length(vec2 const& v) { return sqrt(v.x * v.x + v.y * v.y); }

float length(vec3 const& v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
vec3 norm(vec3 v) { return v / length(v); }
float dot(vec3 const& a, vec3 const& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
vec3 abs(vec3 const& v) { return vec3(fabs(v.x), fabs(v.y), fabs(v.z)); }
vec3 sign(vec3 const& v) { return vec3(sign(v.x), sign(v.y), sign(v.z)); }
vec3 step(vec3 const& edge, vec3 v) { return vec3(step(edge.x, v.x), step(edge.y, v.y), step(edge.z, v.z)); }
vec3 reflect(vec3 rd, vec3 n) { return rd - n * (2 * dot(n, rd)); }

vec3 rotateX(vec3 a, double angle)
{
	vec3 b = a;
	b.z = a.z * cos(angle) - a.y * sin(angle);
	b.y = a.z * sin(angle) + a.y * cos(angle);
	return b;
}

vec3 rotateY(vec3 a, double angle)
{
	vec3 b = a;
	b.x = a.x * cos(angle) - a.z * sin(angle);
	b.z = a.x * sin(angle) + a.z * cos(angle);
	return b;
}

vec3 rotateZ(vec3 a, double angle)
{
	vec3 b = a;
	b.x = a.x * cos(angle) - a.y * sin(angle);
	b.y = a.x * sin(angle) + a.y * cos(angle);
	return b;
}

//ro - sphere position, rd - ray, r - radius
vec2 sphere(vec3 ro, vec3 rd, float r) {
	float b = dot(ro, rd);
	float c = dot(ro, ro) - r * r;
	float h = b * b - c;
	if (h < 0.0) return vec2(-1.0);
	h = sqrt(h);
	return vec2(-b - h, -b + h);
}

//ro - box pos, rd - ray
vec2 box(vec3 ro, vec3 rd, vec3 boxSize, vec3& outNormal) {
	vec3 m = vec3(1.0) / rd;
	vec3 n = m * ro;
	vec3 k = abs(m) * boxSize;
	vec3 t1 = -n - k;
	vec3 t2 = -n + k;
	float tN = fmax(fmax(t1.x, t1.y), t1.z);
	float tF = fmin(fmin(t2.x, t2.y), t2.z);
	if (tN > tF || tF < 0.0) return vec2(-1.0);
	vec3 yzx = vec3(t1.y, t1.z, t1.x);
	vec3 zxy = vec3(t1.z, t1.x, t1.y);
	outNormal = -sign(rd) * step(yzx, t1) * step(zxy, t1);
	return vec2(tN, tF);
}

//ro - camera, rd - ray, p - normal of plane, w - dist from (0,0,0)
float plane(vec3 ro, vec3 rd, vec3 p, float w) {
	return -(dot(ro, p) + w) / dot(rd, p);
}

void SetWindow(int Width, int Height)
{
	_COORD coord;
	coord.X = Width;
	coord.Y = Height;
	_SMALL_RECT Rect;
	Rect.Top = 0;
	Rect.Left = 0;
	Rect.Bottom = Height - 1;
	Rect.Right = Width - 1;
	HANDLE Handle = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleScreenBufferSize(Handle, coord);
	SetConsoleWindowInfo(Handle, TRUE, &Rect);
}

bool Pressed(int key)
{
	return GetKeyState(key) < 0;
}

bool NotInGame()
{
	return ((Pressed(VK_LMENU) && Pressed(VK_TAB)) || Pressed(VK_LWIN) || (Pressed(VK_RMENU) && Pressed(VK_TAB)));
}

int main() {
	float x = 0, y = 3, z = -2;
	//std::cin >> x >> y >> z;
	int width = 120 * 2;
	int height = 30 * 2;
	SetWindow(width, height);
	float aspect = (float)width / height;
	float pixelAspect = 11.0f / 24.0f;
	char gradient[] = " .:!/r(l1Z4H9W8$@";
	int gradientSize = std::size(gradient) - 2;

	wchar_t* screen = new wchar_t[width * height];
	HANDLE hConsole = CreateConsoleScreenBuffer(GENERIC_READ | GENERIC_WRITE, 0, NULL, CONSOLE_TEXTMODE_BUFFER, NULL);
	SetConsoleActiveScreenBuffer(hConsole);
	DWORD dwBytesWritten = 0;

	int h = GetSystemMetrics(SM_CXSCREEN), w = GetSystemMetrics(SM_CYSCREEN);
	SetCursorPos(w / 2, h / 2);
	ShowCursor(FALSE);
	POINT cursorPos{};
	
	int t = 0, t1 = 0;
	float zv = -0.5f, xv = 0.0f, yv = 0.0f, jumph = 0.0f, deltjump = 0.01f;
	bool jump = 0;

	while (1) {
		if (NotInGame()) {
			Sleep(1000);
			while (!Pressed(VK_LCONTROL) ||	!Pressed('R')) {}
			SetCursorPos(w / 2, h / 2);
		}
		GetCursorPos(&cursorPos);
		SetCursorPos(w / 2, h / 2);
		t += cursorPos.x - w / 2;
		t1 += cursorPos.y - h / 2;
		vec3 light = norm(vec3(-0.25, 0.25, -1.0));
		vec3 spherePos = vec3(x, y, z);

		vec3 forw(0, 0, 0);
		if (Pressed('W'))
			forw.x += 0.03;
		if (Pressed('S'))
			forw.x -= 0.03;
		if (Pressed('A'))
			forw.y -= 0.03;
		if (Pressed('D'))
			forw.y += 0.03;
		if (Pressed(VK_LCONTROL))
			zv = -0.05f;
		else
			zv = -0.5f;
		if (Pressed(VK_SPACE) && jump == 0) {
			jump = 1;
			jumph += deltjump;
		}
		if (jump)
		{
			if (jumph < 1.5f && jumph > 0.0f)
				jumph += deltjump;
			else {
				deltjump *= -1;
				if (jumph > 0.0f)
					jumph += deltjump;
				else
					jump = 0;
			}
		}
		if (forw.x != 0.0f && forw.y != 0.0f)
		{
			forw.x *= 0.70710678118f;
			forw.y *= 0.70710678118f;
		}
		zv -= jumph;
		forw = rotateZ(forw, t * 0.003);
		xv += forw.x;
		yv += forw.y;

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				vec2 uv = vec2(i, j) / vec2(width, height) * 2.0f - 1.0f;
				uv.x *= aspect * pixelAspect;
				vec3 ro = vec3(xv, yv, zv);
				vec3 rd = norm(vec3(2, uv));
				rd = rotateY(rd, t1 * 0.003);
				rd = rotateY(rd, 0.25);
				rd = rotateZ(rd, t * 0.003);
				float diff = 1;
				for (int k = 0; k < 50; k++) {
					float minIt = 1000;
					vec2 intersection = sphere(ro - spherePos, rd, 1);
					vec3 n = 0;
					float albedo = 1;
					if (intersection.x > 0) {
						vec3 itPoint = ro - spherePos + rd * intersection.x;
						minIt = intersection.x;
						n = norm(itPoint);
					}
					vec3 boxN = 0;
					intersection = box(ro + vec3(-2, 2, 1), rd, vec3(1, 2, 3), boxN);
					if (intersection.x > 0 && intersection.x < minIt) {
						minIt = intersection.x;
						n = boxN;
					}
					intersection = plane(ro, rd, vec3(0, 0, -1), 1);
					if (intersection.x > 0 && intersection.x < minIt) {
						minIt = intersection.x;
						n = vec3(0, 0, -1);
						albedo = 0.2f;
					}
					intersection = plane(ro, rd, vec3(0, 1, 0), 40);
					if (intersection.x > 0 && intersection.x < minIt) {
						minIt = intersection.x;
						n = vec3(0, 1, 0);
						albedo = 0.2f;
					}
					intersection = plane(ro, rd, vec3(0, 1, 0), -40);
					if (intersection.x > 0 && intersection.x < minIt) {
						minIt = intersection.x;
						n = vec3(0, 1, 0);
						albedo = 0.2f;
					}
					intersection = sphere(ro - vec3(xv,yv,zv), rd, 0.5f);
					if (intersection.x > 0 && intersection.x < minIt) {
						vec3 itPoint = ro + rd * intersection.x;
						minIt = intersection.x;
						n = norm(itPoint);
					}
					if (minIt < 1000) {
						diff *= (dot(n, light) * 0.5 + 0.5) * albedo;
						ro = ro + rd * (minIt - 0.01);
						if (albedo != 0.2f)
							rd = reflect(rd, n);
						else
							break;
					}
					else break;
				}
				int color = (int)(diff * 20);
				color = clamp(color, 0, gradientSize);
				char pixel = gradient[color];
				screen[i + j * width] = pixel;
			}
		}
		screen[width * height - 1] = '\0';
		WriteConsoleOutputCharacterW(hConsole, screen, width * height, { 0, 0 }, &dwBytesWritten);
	}
}