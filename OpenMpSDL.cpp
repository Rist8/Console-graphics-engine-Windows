#include <SDL.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

//==============================================================================
// Math Utilities
//==============================================================================

struct Vec2 { float x, y; Vec2(float x = 0, float y = 0) : x(x), y(y) {} };
struct Vec3 { float x, y, z; Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {} };

inline Vec3 operator+(const Vec3& a, const Vec3& b) { return{ a.x + b.x,a.y + b.y,a.z + b.z }; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return{ a.x - b.x,a.y - b.y,a.z - b.z }; }
inline Vec3 operator*(const Vec3& a, const Vec3& b) { return{ a.x * b.x,a.y * b.y,a.z * b.z }; }
inline Vec3 operator*(const Vec3& v, float s) { return{ v.x * s,v.y * s,v.z * s }; }
inline Vec3 operator*(float s, const Vec3& v) { return{ v.x * s,v.y * s,v.z * s }; }
inline Vec3 operator/(const Vec3& v, float s) { return{ v.x / s,v.y / s,v.z / s }; }
inline Vec3 operator-(const Vec3& v) { return{ -v.x,-v.y,-v.z }; }
inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }
inline Vec3 normalize(const Vec3& v) { float l = length(v); return l > 1e-6f ? v / l : Vec3(); }
inline Vec3 reflect(const Vec3& d, const Vec3& n) { return d - n * (2.0f * dot(d, n)); }

template<typename T>
inline T clamp(T x, T lo, T hi) { if (x < lo) x = lo; if (x > hi) x = hi; return x; }

inline Vec3 rotateX(const Vec3& v, float a) { float c = std::cos(a), s = std::sin(a); return { v.x, v.y * c - v.z * s, v.y * s + v.z * c }; }
inline Vec3 rotateY(const Vec3& v, float a) { float c = std::cos(a), s = std::sin(a); return { v.x * c + v.z * s, v.y, -v.x * s + v.z * c }; }
inline Vec3 rotateZ(const Vec3& v, float a) { float c = std::cos(a), s = std::sin(a); return { v.x * c - v.y * s, v.x * s + v.y * c, v.z }; }

//==============================================================================
// Ray, Intersection and Shapes
//==============================================================================

struct Ray { Vec3 origin, dir; Ray(const Vec3& o, const Vec3& d) : origin(o), dir(normalize(d)) {} };

struct Intersection { float t = -1; Vec3 normal; float albedo = 1.0f; };

class Shape {
public:
    virtual ~Shape() = default;
    virtual Intersection intersect(const Ray& ray) const = 0;
    virtual bool contains(const Vec3& p) const = 0;
};

class Sphere : public Shape {
public:
    Vec3 center;
    float radius;
    float albedo;
    Sphere(const Vec3& c, float r, float a = 1.0f) :center(c), radius(r), albedo(a) {}
    Intersection intersect(const Ray& ray) const override {
        Vec3 oc = ray.origin - center;
        float b = dot(oc, ray.dir);
        float c0 = dot(oc, oc) - radius * radius;
        float disc = b * b - c0;
        if (disc < 0) return {};
        float sq = std::sqrt(disc);
        float t0 = -b - sq, t1 = -b + sq;
        float t = t0 > 0 ? t0 : (t1 > 0 ? t1 : -1);
        if (t < 0) return {};
        Vec3 p = ray.origin + ray.dir * t;
        Vec3 n = normalize(p - center);
        return { t, n, albedo };
    }
    bool contains(const Vec3& p) const override { return length(p - center) < radius; }
};

class Box : public Shape {
public:
    Vec3 center, halfSize;
    float albedo;
    Box(const Vec3& c, const Vec3& s, float a = 1.0f) :center(c), halfSize(s * 0.5f), albedo(a) {}
    Intersection intersect(const Ray& ray) const override {
        Vec3 inv{ 1 / ray.dir.x,1 / ray.dir.y,1 / ray.dir.z };
        Vec3 t1 = (center - halfSize - ray.origin) * inv;
        Vec3 t2 = (center + halfSize - ray.origin) * inv;
        Vec3 tmin{ std::min(t1.x,t2.x),std::min(t1.y,t2.y),std::min(t1.z,t2.z) };
        Vec3 tmax{ std::max(t1.x,t2.x),std::max(t1.y,t2.y),std::max(t1.z,t2.z) };
        float t_near = std::max({ tmin.x,tmin.y,tmin.z });
        float t_far = std::min({ tmax.x,tmax.y,tmax.z });
        if (t_near > t_far || t_far < 0) return {};
        Vec3 n;
        if (t_near == tmin.x) n = { (ray.dir.x < 0.0f) ? 1.0f : -1.0f,0.0f,0.0f };
        else if (t_near == tmin.y) n = { 0.0f,(ray.dir.y < 0.0f) ? 1.0f : -1.0f,0.0f };
        else n = { 0.0f,0.0f,(ray.dir.z < 0.0f) ? 1.0f : -1.0f };
        return { t_near,n,albedo };
    }
    bool contains(const Vec3& p) const override {
        Vec3 d = p - center;
        return std::fabs(d.x) <= halfSize.x && std::fabs(d.y) <= halfSize.y && std::fabs(d.z) <= halfSize.z;
    }
};

class Plane : public Shape {
public:
    Vec3 normal;
    float distance;
    float albedo;
    Plane(const Vec3& n, float d, float a = 0.2f) :normal(normalize(n)), distance(d), albedo(a) {}
    Intersection intersect(const Ray& ray) const override {
        float denom = dot(ray.dir, normal);
        if (std::fabs(denom) < 1e-6f) return {};
        float t = -(dot(ray.origin, normal) + distance) / denom;
        if (t < 0) return {};
        return { t,normal,albedo };
    }
    bool contains(const Vec3& p) const override { return dot(p, normal) + distance <= 0; }
};

class Scene {
public:
    std::vector<Shape*> objects;
    Vec3 lightDir;
    Scene() :lightDir(normalize({ -0.25f,0.25f,-1.0f })) {}
    ~Scene() { for (auto o : objects) delete o; }
    Intersection trace(const Ray& ray) const {
        Intersection best;
        for (auto o : objects) {
            if (o->contains(ray.origin)) continue;
            Intersection hit = o->intersect(ray);
            if (hit.t > 0 && (best.t < 0 || hit.t < best.t)) best = hit;
        }
        return best;
    }
};

//==============================================================================
// SDL2 Renderer & Main
//==============================================================================

int main(int argc, char** argv) {
    const int W = 1920, H = 1080;
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return -1;
    SDL_SetRelativeMouseMode(SDL_TRUE);
    SDL_Window* win = SDL_CreateWindow("SDL Ray Marcher", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W, H, 0);
    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, W, H);
    std::vector<Uint32> pixels(W * H);

    Scene scene;
    scene.objects.push_back(new Sphere({ 0,3,-2 }, 1.0f));
    scene.objects.push_back(new Box({ -2,2,1 }, { 2,4,6 }));
    scene.objects.push_back(new Sphere({ 0,0,0 }, 0.5f));
    scene.objects.push_back(new Plane({ 0,0,-1 }, 1.0f, 0.5f));
    // Uncomment for floor+ceiling
    // scene.objects.push_back(new Plane({0,1,0}, 40.0f));
    // scene.objects.push_back(new Plane({0,1,0}, -40.0f));

    Vec3 camPos{ 0,0,-0.5f }; Vec2 rot{ 0,0 };
    float verticalVel = 0.0f;
    bool running = true;
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_MOUSEMOTION) {
                rot.x += e.motion.yrel * 0.002f;
                rot.y += e.motion.xrel * 0.002f;
            }
        }
        const Uint8* ks = SDL_GetKeyboardState(NULL);
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;

        // Movement
        Vec3 move{ 0,0,0 };
        if (ks[SDL_SCANCODE_W]) move.x += 1;
        if (ks[SDL_SCANCODE_S]) move.x -= 1;
        if (ks[SDL_SCANCODE_A]) move.y -= 1;
        if (ks[SDL_SCANCODE_D]) move.y += 1;
        float mlen = std::sqrt(move.x * move.x + move.y * move.y);
        if (mlen > 0) {
            move.x /= mlen; move.y /= mlen;
            Vec3 fwd{ std::cos(rot.y), std::sin(rot.y), 0 };
            Vec3 right{ -std::sin(rot.y), std::cos(rot.y), 0 };
            camPos = camPos + (fwd * move.x + right * move.y) * (5.0f * dt);
        }
        // Jump & gravity
        if (ks[SDL_SCANCODE_SPACE] && camPos.z <= -0.5f) verticalVel = -5.0f;
        verticalVel += 9.8f * dt;
        camPos.z += verticalVel * dt;
        if (camPos.z > -0.5f) { camPos.z = -0.5f; verticalVel = 0; }

        // Collision resolution
        const float playerRadius = 1.0f;
        const float padding = 0.05f;
        for (auto o : scene.objects) {
            if (auto sp = dynamic_cast<Sphere*>(o)) {
                Vec3 d = camPos - sp->center; float dist = length(d);
                float r = sp->radius + playerRadius + padding;
                if (dist < r) {
                    camPos = sp->center + normalize(d) * r;
                    verticalVel = 0;
                }
            }
            else if (auto bx = dynamic_cast<Box*>(o)) {
                Vec3 delta = camPos - bx->center;
                Vec3 hi = bx->halfSize + Vec3(playerRadius + padding, playerRadius + padding, playerRadius + padding);
                if (std::fabs(delta.x) <= hi.x && std::fabs(delta.y) <= hi.y && std::fabs(delta.z) <= hi.z) {
                    float ox = hi.x - std::fabs(delta.x);
                    float oy = hi.y - std::fabs(delta.y);
                    float oz = hi.z - std::fabs(delta.z);
                    if (ox <= oy && ox <= oz) camPos.x += (delta.x < 0 ? -ox : ox);
                    else if (oy <= ox && oy <= oz) camPos.y += (delta.y < 0 ? -oy : oy);
                    else { camPos.z += (delta.z < 0 ? -oz : oz); verticalVel = 0; }
                }
            }
            else if (auto pl = dynamic_cast<Plane*>(o)) {
                float d = dot(camPos, pl->normal) + pl->distance + playerRadius + padding;
                if (d < 0) { camPos = camPos - pl->normal * d; verticalVel = 0; }
            }
        }

        // Add player sphere for reflection
        Sphere* playerVis = new Sphere(camPos, playerRadius);
        scene.objects.push_back(playerVis);

        // Ray-trace into pixels
        const char* grad = " .:!/r(l1Z4H9W8$@";
        int gsz = int(std::strlen(grad)) - 1;
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float aspect = W / float(H);
                Vec2 uv{ (2.0f * x / W - 1) * aspect, 2.0f * y / H - 1 };
                Vec3 dir = normalize({ 2.0f, uv.x, uv.y });
                dir = rotateX(dir, 0);
                dir = rotateY(dir, -rot.x);
                dir = rotateZ(dir, rot.y);
                Ray ray(camPos, dir);
                float diff = 1.0f;
                Vec3 ori = camPos;
                for (int depth = 0; depth < 20; ++depth) {
                    Intersection hit = scene.trace(Ray(ori, dir));
                    if (hit.t < 0) break;
                    diff *= (dot(hit.normal, scene.lightDir) * 0.5f + 0.5f) * hit.albedo;
                    ori = ori + dir * (hit.t - 0.01f);
                    if (hit.albedo != 0.2f) dir = reflect(dir, hit.normal);
                    else break;
                }
                int c = clamp(int(diff * 20), 0, gsz);
                Uint8 v = Uint8(255 * (c / (float)gsz));
                pixels[x + y * W] = (v << 24) | (v << 16) | (v << 8) | 255;
            }
        }
        scene.objects.pop_back();
        delete playerVis;

        // Render
        SDL_UpdateTexture(tex, NULL, pixels.data(), W * sizeof(Uint32));
        SDL_RenderClear(ren);
        SDL_RenderCopy(ren, tex, NULL, NULL);
        SDL_RenderPresent(ren);
    }

    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
