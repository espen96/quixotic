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
uniform vec2 ScreenSize;
out vec3 chunkOffset;

out float vertexDistance;
out float lm;
out vec4 lm2;
out float lmx;
out float lmy;
out vec4 vertexColor;
out vec2 texCoord0;
out vec2 texCoord2;
out vec2 texCoord3;
out vec4 normal;
out vec4 test;
out vec4 glpos;

#define WAVY_PLANTS
#define WAVY_STRENGTH 0.2 //[0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0]
#define WAVY_SPEED 1000.25 //[0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.25 1.5 2.0 3.0 4.0]
const float PI48 = 150.796447372*WAVY_SPEED;
    float animation = GameTime;
float pi2wt = PI48*animation;

vec2 calcWave(in vec3 pos) {

    float magnitude = abs(sin(dot(vec4(animation, pos),vec4(1.0,0.005,0.005,0.005)))*0.5+0.72);
	vec2 ret = (sin(pi2wt*vec2(0.0063,0.0015)*4. - pos.xz + pos.y*0.05)+0.1)*magnitude;

    return ret;
}

vec3 calcMovePlants(in vec3 pos) {
    vec2 move1 = calcWave(pos );
	float move1y = -length(move1);
   return vec3(move1.x,move1y,move1.y)*5.*WAVY_STRENGTH;
}



void main() {
    vec3 position = Position + ChunkOffset;
    float animation = GameTime * 4000.0;

    float xs = 0.0;
    float ys = 0.0;
    float zs = 0.0;
    if(texture(Sampler0, UV0).a * 255 <= 18.0 && texture(Sampler0, UV0).a*255 >= 17.0) {
            xs = calcMovePlants(position).x;
            ys = calcMovePlants(position).y;
            zs = calcMovePlants(position).z;
        }

    vertexDistance = length((ModelViewMat * vec4(Position + ChunkOffset, 1.0)).xyz);
    vertexColor = Color;
    texCoord0 = UV0;
    texCoord2 = UV2;
    texCoord3 = vec2(0.0);

    
    lm = clamp((float(UV2.y)/255)-(float(UV2.x)/255),0,1);
    lmx = clamp((float(UV2.y)/255),0,1);
    lmy = clamp((float(UV2.x)/255),0,1);


    lm2 = minecraft_sample_lightmap2(Sampler2, UV2);
    normal = normalize(ModelViewMat * vec4(Normal, 0.0));

    xs *= lmx;
    zs *= lmx;
    gl_Position = ProjMat * ModelViewMat * (vec4(position, 1.0) + vec4(xs / 32.0, ys / 32.0, zs / 32.0, 0.0));
    glpos = gl_Position;

}
