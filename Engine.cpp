#include <windows.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

//==============================================================================
// Math Utilities
//==============================================================================

#undef min
#undef max

struct Vec2 {
    float x, y;
    Vec2(float x = 0, float y = 0) : x(x), y(y) {}
};

struct Vec3 {
    float x, y, z;
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
};

// Vec3 arithmetic
inline Vec3 operator+(const Vec3& a, const Vec3& b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline Vec3 operator*(const Vec3& a, const Vec3& b) { return { a.x * b.x, a.y * b.y, a.z * b.z }; }
inline Vec3 operator*(const Vec3& v, float s) { return { v.x * s,   v.y * s,   v.z * s }; }
inline Vec3 operator*(float s, const Vec3& v) { return { v.x * s,   v.y * s,   v.z * s }; }
inline Vec3 operator/(const Vec3& v, float s) { return { v.x / s,   v.y / s,   v.z / s }; }
inline Vec3 operator-(const Vec3& v) { return { -v.x, -v.y, -v.z }; }

// Scalar clamp
template <typename T>
inline T clamp(T x, T lo, T hi) {
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    return x;
}

inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(const Vec3& v) { float len = length(v); return len > 1e-6f ? v / len : Vec3(); }
inline Vec3 reflect(const Vec3& dir, const Vec3& n) { return dir - n * (2.0f * dot(dir, n)); }

//==============================================================================
// Rotation Helpers
//==============================================================================

inline Vec3 rotateX(const Vec3& v, float a) {
    float c = std::cos(a), s = std::sin(a);
    return { v.x, v.y * c - v.z * s, v.y * s + v.z * c };
}
inline Vec3 rotateY(const Vec3& v, float a) {
    float c = std::cos(a), s = std::sin(a);
    return { v.x * c + v.z * s, v.y, -v.x * s + v.z * c };
}
inline Vec3 rotateZ(const Vec3& v, float a) {
    float c = std::cos(a), s = std::sin(a);
    return { v.x * c - v.y * s, v.x * s + v.y * c, v.z };
}

//==============================================================================
// Ray, Intersection and Material
//==============================================================================

struct Ray {
    Vec3 origin;
    Vec3 dir;
    Ray(const Vec3& o, const Vec3& d) : origin(o), dir(normalize(d)) {}
};

struct Intersection {
    float t = -1;
    Vec3 normal;
    float albedo = 1.0f;
};

//==============================================================================
// Abstract Shape
//==============================================================================

class Shape {
public:
    virtual ~Shape() = default;
    virtual Intersection intersect(const Ray& ray) const = 0;
    virtual bool contains(const Vec3& p) const = 0;
};

//==============================================================================
// Sphere Shape
//==============================================================================

class Sphere : public Shape {
public:
    Vec3 center;
    float radius;
    Sphere(const Vec3& c, float r) : center(c), radius(r) {}
    Intersection intersect(const Ray& ray) const override {
        Vec3 oc = ray.origin - center;
        float b = dot(oc, ray.dir);
        float c = dot(oc, oc) - radius * radius;
        float h2 = b * b - c;
        if (h2 < 0) return {};
        float h = std::sqrt(h2);
        float t = -b - h;
        if (t < 0) t = -b + h;
        if (t < 0) return {};
        Vec3 p = ray.origin + ray.dir * t;
        Vec3 n = normalize(p - center);
        return { t, n, 1.0f };
    }
    bool contains(const Vec3& p) const override { return length(p - center) < radius; }
};

//==============================================================================
// Axis-Aligned Box Shape
//==============================================================================

class Box : public Shape {
public:
    Vec3 center;
    Vec3 halfSize;
    Box(const Vec3& c, const Vec3& size) : center(c), halfSize(size * 0.5f) {}
    Intersection intersect(const Ray& ray) const override {
        Vec3 inv = { 1 / ray.dir.x, 1 / ray.dir.y, 1 / ray.dir.z };
        Vec3 t1 = (center - halfSize - ray.origin) * inv;
        Vec3 t2 = (center + halfSize - ray.origin) * inv;
        Vec3 tmin = { std::min(t1.x, t2.x), std::min(t1.y, t2.y), std::min(t1.z, t2.z) };
        Vec3 tmax = { std::max(t1.x, t2.x), std::max(t1.y, t2.y), std::max(t1.z, t2.z) };
        float t_near = std::max({ tmin.x, tmin.y, tmin.z });
        float t_far = std::min({ tmax.x, tmax.y, tmax.z });
        if (t_near > t_far || t_far < 0) return {};
        Vec3 normal;
        if (t_near == tmin.x) normal = { (ray.dir.x < 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f };
        else if (t_near == tmin.y) normal = { 0.0f, (ray.dir.y < 0.0f) ? 1.0f : -1.0f, 0.0f };
        else normal = { 0.0f, 0.0f, (ray.dir.z < 0.0f) ? 1.0f : -1.0f };
        return { t_near, normal, 1.0f };
    }
    bool contains(const Vec3& p) const override {
        Vec3 d = p - center; return fabs(d.x) <= halfSize.x && fabs(d.y) <= halfSize.y && fabs(d.z) <= halfSize.z;
    }
};

//==============================================================================
// Infinite Plane Shape
//==============================================================================

class Plane : public Shape {
public:
    Vec3 normal;
    float distance;
    float albedo;
    Plane(const Vec3& n, float d, float a = 0.2f)
        : normal(normalize(n)), distance(d), albedo(a) {
    }
    Intersection intersect(const Ray& ray) const override {
        float denom = dot(ray.dir, normal);
        if (std::fabs(denom) < 1e-6f) return {};
        float t = -(dot(ray.origin, normal) + distance) / denom;
        if (t < 0) return {};
        return { t, normal, albedo };
    }
    bool contains(const Vec3& p) const override {
        return dot(p, normal) + distance <= 0;
    }
};

//==============================================================================
// Scene with Objects and Light
//==============================================================================

class Scene {
public:
    std::vector<Shape*> objects;
    Vec3 lightDir;
    Scene() : lightDir(normalize({ -0.25f, 0.25f, -1.0f })) {}
    ~Scene() { for (auto o : objects) delete o; }
    Intersection trace(const Ray& ray) const {
        Intersection best;
        for (auto o : objects) {
            if (o->contains(ray.origin)) continue;
            Intersection i = o->intersect(ray);
            if (i.t > 0 && (best.t < 0 || i.t < best.t)) best = i;
        }
        return best;
    }
};

//==============================================================================
// Renderer
//==============================================================================

class Renderer {
    int width, height;
    float aspect, pixelAspect;
    std::vector<wchar_t> buffer;
    HANDLE console;
public:
    Renderer(int w, int h)
        : width(w), height(h)
        , aspect((float)w / h), pixelAspect(11.0f / 24.0f)
        , buffer(w* h, L' ')
    {
        console = CreateConsoleScreenBuffer(GENERIC_READ | GENERIC_WRITE, 0, NULL, CONSOLE_TEXTMODE_BUFFER, NULL);
        SetConsoleActiveScreenBuffer(console);
    }

    void renderFrame(const Scene& scene, const Vec3& camPos, const Vec2& rot) {
        static const char grad[] = " .:!/r(l1Z4H9W8$@";
        const int gsz = sizeof(grad) - 2;

#pragma omp.parallel for schedule(dynamic)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Vec2 uv = { (2.0f * x / width - 1) * aspect * pixelAspect, 2.0f * y / height - 1 };
                Vec3 dir = normalize({ 2.0f, uv.x, uv.y });
                dir = rotateX(dir, 0.0f);
                dir = rotateY(dir, -rot.x);
                dir = rotateZ(dir, rot.y);  // roll angle here

                Ray ray(camPos, dir);
                float diff = 1.0f;
                Vec3 origin = camPos;
                for (int depth = 0; depth < 50; ++depth) {
                    Intersection h = scene.trace(Ray(origin, dir));
                    if (h.t < 0) break;
                    diff *= (dot(h.normal, scene.lightDir) * 0.5f + 0.5f) * h.albedo;
                    origin = origin + dir * (h.t - 0.01f);
                    if (h.albedo != 0.2f) dir = reflect(dir, h.normal);
                    else break;
                }

                int c = clamp(int(diff * 20), 0, gsz);
                buffer[x + y * width] = grad[c];
            }
        }
        DWORD written;
        WriteConsoleOutputCharacterW(console, buffer.data(), buffer.size(), { 0,0 }, &written);
    }
};

//==============================================================================
// Input & Application Entry
//==============================================================================

bool keyDown(int vk) { return (GetAsyncKeyState(vk) & 0x8000) != 0; }

int main() {
    const int WIDTH = 240, HEIGHT = 60;
    Vec3 camPos(0, 0, -0.5f);
    Vec2 rotation(0, 0);
    float speed = 5.0f;
    const float playerRadius = 1.0f, padding = 0.05f;

    Renderer renderer(WIDTH, HEIGHT);
    Scene scene;
    scene.objects.push_back(new Sphere({ 0, 3, -2 }, 1.0f));
    scene.objects.push_back(new Box({ -2, 2, 1 }, { 2, 4, 6 }));
    scene.objects.push_back(new Sphere({ 0, 0, 0 }, 0.5f));
    scene.objects.push_back(new Plane({ 0, 0, -1 }, 1.0f));
    //scene.objects.push_back(new Plane({ 0, 1, 0 }, 40.0f));
    //scene.objects.push_back(new Plane({ 0, 1, 0 }, -40.0f));

    CONSOLE_CURSOR_INFO cursorInfo{ 1, FALSE };
    SetConsoleCursorInfo(GetStdHandle(STD_OUTPUT_HANDLE), &cursorInfo);

    POINT center{ GetSystemMetrics(SM_CXSCREEN) / 2, GetSystemMetrics(SM_CYSCREEN) / 2 };
    SetCursorPos(center.x, center.y);
    ShowCursor(FALSE);

    auto lastTime = std::chrono::high_resolution_clock::now();
    static float verticalVel = 0.0f;
    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;

        POINT pt; GetCursorPos(&pt);
        rotation.x += (pt.y - center.y) * 0.002f;
        rotation.y += (pt.x - center.x) * 0.002f;
        SetCursorPos(center.x, center.y);

        Vec3 move(0, 0, 0);
        if (keyDown('W')) move.x += 1;
        if (keyDown('S')) move.x -= 1;
        if (keyDown('A')) move.y -= 1;
        if (keyDown('D')) move.y += 1;
        float mlen = std::sqrt(move.x * move.x + move.y * move.y);
        if (mlen > 0) {
            move.x /= mlen; move.y /= mlen;
            Vec3 forward = { std::cos(rotation.y), std::sin(rotation.y), 0 };
            Vec3 right = { -std::sin(rotation.y), std::cos(rotation.y), 0 };
            camPos = camPos + (forward * move.x + right * move.y) * (speed * dt);
        }

        // Jump & gravity
        if (keyDown(VK_SPACE) && camPos.z <= -0.5f) verticalVel = -5.0f;
        verticalVel += 9.8f * dt;
        camPos.z += verticalVel * dt;
        if (camPos.z > -0.5f) { camPos.z = -0.5f; verticalVel = 0.0f; }

        for (auto obj : scene.objects) {
            if (auto sph = dynamic_cast<Sphere*>(obj)) {
                Vec3 diff = camPos - sph->center; float d = length(diff);
                float r = sph->radius + playerRadius + padding;
                if (d < r) { camPos = sph->center + normalize(diff) * r; verticalVel = 0; }
            }
            else if (auto bx = dynamic_cast<Box*>(obj)) {
                Vec3 delta = camPos - bx->center;
                Vec3 hi = bx->halfSize + Vec3(playerRadius + padding, playerRadius + padding, playerRadius + padding);
                if (std::fabs(delta.x) <= hi.x && std::fabs(delta.y) <= hi.y && std::fabs(delta.z) <= hi.z) {
                    // push out along smallest overlap axis
                    float ox = hi.x - std::fabs(delta.x);
                    float oy = hi.y - std::fabs(delta.y);
                    float oz = hi.z - std::fabs(delta.z);
                    if (ox <= oy && ox <= oz) { camPos.x += (delta.x < 0 ? -ox : ox); }
                    else if (oy <= ox && oy <= oz) { camPos.y += (delta.y < 0 ? -oy : oy); }
                    else { camPos.z += (delta.z < 0 ? -oz : oz); verticalVel = 0; }
                }
            }
            else if (auto pl = dynamic_cast<Plane*>(obj)) {
                float dist = dot(camPos, pl->normal) + pl->distance + playerRadius + padding;
                if (dist < 0) { camPos = camPos - pl->normal * dist; verticalVel = 0; }
            }
        }

        // Add player sphere at camera position for visibility
        Sphere* playerVis = new Sphere(camPos, playerRadius);
        scene.objects.push_back(playerVis);
        renderer.renderFrame(scene, camPos, rotation);
        scene.objects.pop_back();
        delete playerVis;
    }
    return 0;
}
