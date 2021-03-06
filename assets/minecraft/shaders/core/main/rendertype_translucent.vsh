#version 150

#moj_import <light.glsl>

in vec3 Position;
in vec4 Color;

in vec2 UV0;

in ivec2 UV2;
in vec3 Normal;

uniform sampler2D Sampler0;
uniform sampler2D Sampler2;
uniform float GameTime;

uniform mat4 ModelViewMat;
uniform mat4 ProjMat;
uniform vec3 ChunkOffset;

out float water;
out float wavea;
out vec4 vertexColor;
out vec2 texCoord0;
out vec4 normal;
out vec4 lm;
out vec4 glpos;
out vec3 noise;
out vec3 color2;
out float lmx;
out float lmy;
#define atlasTileDim 1024.0 // Atlas dimensions in texture tiles
#define tileSizePixels 16.0 // Texture tile size in pixels

#define VERTICES_ATLAS_TEXTURE(u, v, x, y) x >= u/atlasTileDim && x <= (u+16)/atlasTileDim && y >= v/atlasTileDim && y <= (v+16)/atlasTileDim

#define VERTICES_WATER_STILL(x, y) VERTICES_ATLAS_TEXTURE(496, 416, x, y)
const float PI = 3.1415927;

float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

float wave(float n) {
    return sin(2 * PI * (n));
}

float waterH(vec3 posxz) {

float wave = 0.0;


float factor = 1.1;
float amplitude = 0.02;
float speed = 4.0;
float size = 0.2;

float px = posxz.x/50.0 + 250.0;
float py = posxz.z/50.0  + 250.0;

float fpx = abs(fract(px*20.0)-0.5)*2.0;
float fpy = abs(fract(py*20.0)-0.5)*2.0;

float d = length(vec2(fpx,fpy));

for (int i = 0; i < 3; i++) {
wave -= d*factor*cos( (1/factor)*px*py*size + 1.0*(GameTime * 500.0)*speed);
factor /= 2;
}

factor = 1.0;
px = -posxz.x/50.0 + 250.0;
py = -posxz.z/150.0 - 250.0;

fpx = abs(fract(px*20.0)-0.5)*2.0;
fpy = abs(fract(py*20.0)-0.5)*2.0;

d = length(vec2(fpx,fpy));
float wave2 = 0.0;
for (int i = 0; i < 3; i++) {
wave2 -= d*factor*cos( (1/factor)*px*py*size + 1.0*(GameTime * 500.0)*speed);
factor /= 2;
}

return amplitude*wave2+amplitude*wave;
}


const vec2 COPRIMES = vec2(2, 3);

vec2 halton(int index) {
    vec2 f = vec2(1);
    vec2 result = vec2(0);
    vec2 ind = vec2(index);

    while(ind.x > 0.0 && ind.y > 0.0) {
        f /= COPRIMES;
        result += f * mod(ind, COPRIMES);
        ind = floor(ind / COPRIMES);
    }
    return result;
}

vec2 calculateJitter() {
    return (halton(int(mod((GameTime * 3.0) * 24000.0, 128))) - 0.5) / 1024.0;
}

void main() {
 //   gl_Position = ProjMat * ModelViewMat * vec4(Position + ChunkOffset, 1.0);

    float wtest = (texture(Sampler0, UV0).a);

	vec3 posxz = mod(Position,16);

	posxz.x += sin(posxz.z+(GameTime * 500.0))*0.25;
	posxz.z += cos(posxz.x+(GameTime * 500.0)*0.5)*0.25;

    lmx = clamp((float(UV2.y) / 255), 0, 1);
    lmy = clamp((float(UV2.x) / 255), 0, 1);
    float modif = halton(int(mod((GameTime * 10.0),128))).x;
    wavea = 0.0;
    if(wtest * 255 == 200)
        wavea = (waterH(posxz) * clamp((float(UV2.y) / 255), 0.1, 1))*0.5;
    vec4 viewPos = ModelViewMat * vec4(Position + vec3(0, wavea, 0) + ChunkOffset, 1.0)+ vec4(calculateJitter()*0.25,0,0);
    gl_Position = ProjMat * viewPos;
    noise = vec3(wavea);
//    vertexDistance = length((ModelViewMat * vec4(Position + ChunkOffset, 1.0)).xyz);
    vertexColor = Color* minecraft_sample_lightmap2(Sampler2, UV2);

    texCoord0 = UV0;
//    normal = ProjMat * ModelViewMat * vec4(Normal, 0.0);
    color2.rgb = vec3(water);
    float test = 0;
    if(posxz.z < 0.5)
        test = 1;
    lm =  texelFetch(Sampler2, UV2 / 16, 0);    

    glpos = vec4(waterH(posxz) * 10);
}
