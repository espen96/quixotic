#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D temporals3Sampler;
uniform vec2 OutSize;
uniform vec2 ScreenSize;
uniform float Time;
uniform mat4 ProjMat;

in vec3 flareColor;

in float GameTime;
in vec2 texCoord;
in vec2 texCoord2;
in vec2 oneTexel;
in vec3 sunDir;
in vec4 fogcol;
in vec4 skycol;
in vec4 rain;
in mat4 gbufferModelViewInverse;
in mat4 gbufferModelView;
in mat4 gbufferProjection;
in mat4 gbufferProjectionInverse;
in float near;
in float far;

in float aspectRatio;
in float cosFOVrad;
in float tanFOVrad;
in mat4 gbPI;
in mat4 gbP;






out vec4 fragColor;

// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define NUMCONTROLS 26
#define THRESH 0.5
#define FPRECISION 4000000.0
#define PROJNEAR 0.05
#define FUDGE 32.0


float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}

vec3 toLinear(vec3 sRGB){
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}
#define SHARPENING 0.1 

void main() {
    float aspectRatio = ScreenSize.x/ScreenSize.y;
    vec4 screenPos = gl_FragCoord;
         screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
         screenPos.zw = vec2(1.0);
    vec3 view = normalize((gbufferModelViewInverse * screenPos).xyz);


    float diffuseDepth = texture(DiffuseDepthSampler, texCoord).r;
    vec3 OutTexel = texture(DiffuseSampler, texCoord).rgb;


    // Get centre location
    vec2 pos = texCoord;
    
    // Init value
    vec4 color1 = vec4(0.0, 0.0, 0.0, 0.0); 
    vec4 color2; // to set color later
    
    // Init kernel and number of steps
    //vec4 kernel = vec4(0.399, 0.242, 0.054, 0.004); // Gaussian sigma 1.0
    //vec4 kernel = vec4(0.53, 0.22, 0.015, 0.00018); // Gaussian sigma 0.75
    vec4 kernel = vec4(0.79, 0.11, 0.0026, 0.000001); // Gaussian sigma 0.5
    int sze = 4; 
    
    // Init step size in tex coords
    float dx = oneTexel.x;
    float dy = oneTexel.y;
    
    // Convolve
    for (int y=-sze; y<sze+1; y++)
    {
        for (int x=-sze; x<sze+1; x++)
        {   
            float k = kernel[int(abs(float(x)))] * kernel[int(abs(float(y)))];
            vec2 dpos = vec2(float(x)*dx, float(y)*dy);
            color1 += texture(DiffuseSampler, pos+dpos) * k;
        }
    }
/*
    //Weights : 1 in the center, 0.5 middle, 0.25 corners
    vec3 albedoCurrent1 = texture(DiffuseSampler, texCoord + vec2(oneTexel.x,oneTexel.y)/1*0.5).rgb;
    vec3 albedoCurrent2 = texture(DiffuseSampler, texCoord + vec2(oneTexel.x,-oneTexel.y)/1*0.5).rgb;
    vec3 albedoCurrent3 = texture(DiffuseSampler, texCoord + vec2(-oneTexel.x,-oneTexel.y)/1*0.5).rgb;
    vec3 albedoCurrent4 = texture(DiffuseSampler, texCoord + vec2(-oneTexel.x,oneTexel.y)/1*0.5).rgb;


    vec3 m1 = -0.5/3.5*color1.rgb + albedoCurrent1/3.5 + albedoCurrent2/3.5 + albedoCurrent3/3.5 + albedoCurrent4/3.5;
    vec3 std = abs(color1.rgb - m1) + abs(albedoCurrent1 - m1) + abs(albedoCurrent2 - m1) +
     abs(albedoCurrent3 - m1) + abs(albedoCurrent3 - m1) + abs(albedoCurrent4 - m1);
    float contrast = 1.0 - luma(std)/5.0;
    color1.rgb = color1.rgb*(1.0+(SHARPENING)*contrast)
          - (SHARPENING)/(1.0-0.5/3.5)*contrast*(m1 - 0.5/3.5*color1.rgb);

*/
    fragColor = color1;

}
