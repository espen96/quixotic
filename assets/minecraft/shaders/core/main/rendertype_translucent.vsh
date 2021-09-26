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




void main() {
 //   gl_Position = ProjMat * ModelViewMat * vec4(Position + ChunkOffset, 1.0);
    vec3 position = Position + ChunkOffset;
    vec3 position2 = Position ;
    float animation = GameTime * 2000.0;
    float animation3 = (GameTime * 4000.0);
    float xs = 0.0;
    float zs = 0.0;
         water = 0.0;
    float offset_y = 0.0;
    float wtest = (  texture(Sampler0, UV0).a);

    if(wtest*255 == 200) {

      
          xs = sin(position.x + animation);
          xs += sin(position.z );
          xs += sin(position.x + animation3);
          xs += sin(position.x + animation3);    
          xs *= sin(position.z *0.5);
          xs *= sin(position.z - 0.8 + animation) + 1.0 * sin((position.z + 0.5) / 3 + animation) + 2.0 * sin((position.z - 20.0) / 10.0 + animation3) + sin(position.z + 30 - animation);


          zs = cos(position.z + animation);
          zs += cos(position.x + animation3);
          zs += cos(position.z + animation);
          zs += sin(position.z*0.5 );
          zs *= sin(position.x - 0.8 + animation3) + 1.0 * sin((position.x + 0.5) / 4 + animation) + 2.0 * sin((position.x - 20.0) / 10.0 + animation) + sin(position.x + 30 - animation);


          xs *= 0.2;
          zs *= 0.2;
          water = 1;

            
    }
    noise = vec3(xs,zs,0);
    gl_Position = ProjMat * ModelViewMat * (vec4(position, 1.0) + vec4(0.0, (xs - zs) / 128.0, 0.0, 0.0));

    vertexDistance = length((ModelViewMat * vec4(Position + ChunkOffset, 1.0)).xyz);
    vertexColor = Color * texelFetch(Sampler2, UV2 / 16, 0);
    
    texCoord0 = UV0;
    normal = ProjMat * ModelViewMat * vec4(Normal, 0.0);
      color2.rgb = vec3(water);
    glpos = gl_Position;
}
