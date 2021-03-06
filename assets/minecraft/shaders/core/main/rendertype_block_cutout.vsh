#version 150

#moj_import <light.glsl>

in vec3 Position;
in vec4 Color;
in vec2 UV0;
in ivec2 UV2;

uniform sampler2D Sampler0;
uniform sampler2D Sampler2;
uniform float GameTime;
out mat4 ProjMat2;
uniform mat4 ModelViewMat;
uniform mat4 ProjMat;
uniform vec3 ChunkOffset;

out float lmx;
out float lmy;
out vec4 vertexColor;
out vec2 texCoord0;
noperspective out vec3 test;
out vec4 glpos;

out vec3 cornerTex1;
out vec3 cornerTex2;
out vec3 cornerTex3;
out vec3 viewPos;


#define WAVY_PLANTS
#define WAVY_STRENGTH 0.3 
#define WAVY_SPEED 12.0 
const float PI48 = 150.796447372 * WAVY_SPEED;
float animation = GameTime;
float pi2wt = PI48 * animation;

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
vec3 wavingLeaves(vec3 viewPos) {
float t = pi2wt;

float magnitude = sin((t * 2.0) + viewPos.x + viewPos.z) * 0.02 + 0.02;

float d0 = sin(t * 0.367867992224) * 3.0 - 2.0;
float d1 = sin(t * 0.295262467443) * 3.0 - 2.0;
float d2 = sin(t * 0.233749453392) * 3.0 - 2.0;
float d3 = sin(t * 0.316055598953) * 3.0 - 2.0;

vec3 wind = vec3(0.0);

wind.x += sin((t * 1.80499344071) + (viewPos.x + d0) * 0.5 + (viewPos.z + d1) * 0.5 + viewPos.y) * magnitude * 0.25;
wind.z += sin((t * 1.49332750285) + (viewPos.y + d2) * 0.5 + (viewPos.x + d3) * 0.5 + viewPos.y) * magnitude * 2.0;
wind.y += sin((t * 0.48798950513) + (viewPos.z + d2) * 0.5 + (viewPos.x + d3) * 0.5 + viewPos.y) * magnitude * 0.5;

return wind;
}

void main() {
vec3 position = Position + ChunkOffset;
test = texture(Sampler0, UV0).rgb;
vec3 wave = vec3(0.0);

if(texture(Sampler0, UV0).a * 255 <= 18.0 && texture(Sampler0, UV0).a * 255 >= 17.0) {
wave = wavingLeaves(mod(Position,16)).xyz;

}
vertexColor = Color;
texCoord0 = UV0;
ProjMat2 = ProjMat;

lmx = clamp((float(UV2.y) / 255), 0, 1);
lmy = clamp((float(UV2.x) / 255), 0, 1);
float vertexDistance = length((ModelViewMat * vec4(Position + ChunkOffset, 1.0)).xyz);
if(vertexDistance > 64) {
wave = vec3(0.0);

}

gl_Position = ProjMat * ModelViewMat * (vec4(position, 1.0) + vec4(wave * lmx, 0.0) + vec4(calculateJitter() * 0.0, 0, 0));
glpos = gl_Position;
    cornerTex1 = vec3(0.0);
    cornerTex2 = vec3(0.0);
    cornerTex3 = vec3(0.0);
    if (gl_VertexID % 4 == 0) cornerTex1 = vec3(UV0, 1.0);
    if (gl_VertexID % 4 == 2) cornerTex2 = vec3(UV0, 1.0);
    if (gl_VertexID % 2 == 1) cornerTex3 = vec3(UV0, 1.0);
    viewPos = (ModelViewMat * vec4(position, 1.0)).xyz;



}
