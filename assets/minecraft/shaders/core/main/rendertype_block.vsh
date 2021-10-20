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
out vec3 chunkOffset;
out vec4 normal;
out float vertexDistance;
out float lm;
out float lightf;
out vec4 lm2;
out float lmx;
out float lmy;
out vec4 vertexColor;
noperspective out vec3 test;
out vec2 texCoord0;
out vec2 texCoord2;

out vec4 glpos;

const vec2 COPRIMES = vec2(2, 3);

vec2 halton(int index) {
    vec2 f = vec2(1);
    vec2 result = vec2(0);
    vec2 ind = vec2(index);

    while (ind.x > 0.0 && ind.y > 0.0) {
        f /= COPRIMES;
        result += f * mod(ind, COPRIMES);
        ind = floor(ind / COPRIMES);
    }
    return result;
}

vec2 calculateJitter() {
    return (halton(int(mod((GameTime*3.0) * 24000.0, 128))) - 0.5) / 1024.0;
}

void main() {
    vec3 position = Position + ChunkOffset;
    float animation = GameTime * 4000.0;
  test = textureLod(Sampler0, UV0,100).rgb ;

    float xs = 0.0;
    float zs = 0.0;
    if(texture(Sampler0, UV0).a * 255 <= 18.0 && texture(Sampler0, UV0).a*255 >= 17.0) {
            xs = sin(position.x + animation);
            zs = cos(position.z + animation);
        }


   // vertexDistance = length((ModelViewMat * vec4(Position + ChunkOffset, 1.0)).xyz);
    vertexColor = Color;
    texCoord0 = UV0;
    texCoord2 = UV2;
    lm = clamp((float(UV2.y)/255)-(float(UV2.x)/255),0,1);
    lmx = clamp((float(UV2.y)/255),0,1);
    lmy = clamp((float(UV2.x)/255),0,1);

    normal = normalize(ModelViewMat * vec4(Normal, 0.0));
    lm2 = minecraft_sample_lightmap2(Sampler2, UV2);
 
    gl_Position = ProjMat * ModelViewMat * (vec4(position, 1.0) + vec4(xs / 32.0, 0.0, zs / 32.0, 0.0)+vec4(calculateJitter()*0, 0, 0));

   glpos = gl_Position;        
}
