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

out float vertexDistance;
out float water;
out vec4 vertexColor;
out vec2 texCoord0;
out vec4 normal;
out vec4 glpos;
out vec3 noise;
out vec3 color2;

#define atlasTileDim 1024.0 // Atlas dimensions in texture tiles
#define tileSizePixels 16.0 // Texture tile size in pixels

#define VERTICES_ATLAS_TEXTURE(u, v, x, y) x >= u/atlasTileDim && x <= (u+16)/atlasTileDim && y >= v/atlasTileDim && y <= (v+16)/atlasTileDim

#define VERTICES_WATER_STILL(x, y) VERTICES_ATLAS_TEXTURE(496, 416, x, y)
const float PI = 3.1415927;


float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}

float wave(float n) {
return sin(2 * PI * (n));
}

float waterH(vec3 posxz) {
posxz *=16;
float wave = 0.0;


float factor = 1.0;
float amplitude = 0.01;
float speed = 4.0;
float size = 0.1;

float px = posxz.x/50.0 + 250.0;
float py = posxz.z/50.0  + 250.0;

float fpx = abs(fract(px*20.0)-0.5)*2.0;
float fpy = abs(fract(py*20.0)-0.5)*2.0;

float d = length(vec2(fpx,fpy));

for (int i = 0; i < 3; i++) {
wave -= d*factor*cos( (1/factor)*px*py*size + 1.0*( GameTime * 500.0)*speed);
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
wave2 -= d*factor*cos( (1/factor)*px*py*size + 1.0*( GameTime * 800.0)*speed);
factor /= 2;
}

return amplitude*wave2+amplitude*wave;
}

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
 //   gl_Position = ProjMat * ModelViewMat * vec4(Position + ChunkOffset, 1.0);
    vec3 position = Position + ChunkOffset;
    
    mat4 gbufferModelViewInverse = inverse(ModelViewMat);
   

    vec3 position2 = Position ;
    float animation = GameTime * 1000.0;
    float animation3 = (GameTime * 2000.0);
    float xs = 0.0;
    float zs = 0.0;
         water = 0.0;
    float offset_y = 0.0;
    float wtest = (  texture(Sampler0, UV0).a);
/*
    if(wtest*255 == 200) {

      
          xs = sin(position.x + animation);
          xs += sin(position.x + animation3);
          xs += sin(position.x + animation);    
          xs *= sin(position.z *0.5);
          xs *= sin(position.z - 0.8 + animation) + 1.0 * sin((position.z + 0.5) / 3 + animation) + 2.0 * sin((position.z - 20.0) / 10.0 + animation3) + sin(position.z + 30 - animation);


          zs = cos(position.z + animation);
          zs += cos(position.x + animation);
          zs += cos(position.z + animation3);
          zs += sin(position.z*0.5 );
          zs *= sin(position.x - 0.8 + animation3) + 1.0 * sin((position.x + 0.5) / 4 + animation) + 2.0 * sin((position.x - 20.0) / 10.0 + animation) + sin(position.x + 30 - animation);


          xs *= 0.7 + (fract(GameTime)*0.01);
          zs *= 0.5;
          water = 1;

            
    }	

    */
    vec3 posxz = sin(Position-0.145); 


    noise = vec3(xs,zs,0);
    float wavea = 0.0;
    if(wtest*255 == 200)  wavea = (waterH(posxz)*clamp((float(UV2.y)/255),0.1,1));
    vec4 viewPos = ModelViewMat * vec4(Position+ vec3( 0, wavea,0 ) + ChunkOffset, 1.0)+vec4(calculateJitter()*1.5, 0, 0);
    gl_Position = ProjMat * viewPos;

//    vertexDistance = length((ModelViewMat * vec4(Position + ChunkOffset, 1.0)).xyz);
    vertexColor = Color * texelFetch(Sampler2, UV2 / 16, 0);
    
    texCoord0 = UV0;
//    normal = ProjMat * ModelViewMat * vec4(Normal, 0.0);
      color2.rgb = vec3(water);
      float test = 0;
          if(posxz.z < 0.5) test = 1;

    glpos = vec4(waterH(posxz)*10);
}
