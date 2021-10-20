#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>
#moj_import <mappings.glsl>

uniform sampler2D Sampler0;
uniform sampler2D Sampler2;
uniform vec2 ScreenSize;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;
in vec3 chunkOffset;

in float vertexDistance;
in float lm;
in vec4 vertexColor;
in vec4 lm2;
in vec2 texCoord0;
in vec2 texCoord2;
in vec2 texCoord3;
in vec4 normal;
noperspective in vec3 test;
in vec4 glpos;
in float lmx;
in float lmy;
out vec4 fragColor;


void main() {

  vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );


  discardControlGLPos(gl_FragCoord.xy, glpos);
                  

  vec4 albedo = textureLod(Sampler0, texCoord0,0);
  if(albedo.a >0.5) albedo = texture(Sampler0, texCoord0);
  vec4 color = albedo * vertexColor * ColorModulator;
  float alpha = color.a;
  float lightm = 0;


//  color.rgb = (color.rgb*lm2.rgb);
        
  if (color.a*255 <= 17.0) {
    discard;
  }
    color.rgb +=rnd/16; 
    color.rgb = clamp(color.rgb,0.001,1);

  float translucent = 0;

  float mod2 = gl_FragCoord.x + gl_FragCoord.y;
  float res = mod(mod2, 2.0f);

    float lum = luma4(albedo.rgb);
	vec3 diff = albedo.rgb-lum;

  float alpha0 = int(textureLod(Sampler0, texCoord0,0).a*255);
  if(alpha0==255){
  float procedual1 = ((distance(textureLod(Sampler0, texCoord0,0).rgb,test.rgb)))*255;

//color.rgb = test;
vec3 test2 = floor(test.rgb*255);
float test3  = floor(test2.r+test2.g+test2.b);
//if(vertexColor.g >0.1)alpha0 =30; 
if (diff.r < 0.1 && diff.b < 0.05 && lum< 0.8) alpha0 = int(floor(map((albedo.g*0.5)*255,0,255,sssMin,sssMax)));

 //if(test3 <= 305 && test3 >= 295 && test2.r >= 110 && test2.b <= 90)  alpha0 = clamp(procedual1*albedo.r,lightMin,lightMax);
 //if(test3 <= 255 && test3 >= 250 && test2.r >= 105 && test2.b <= 90)  alpha0 = clamp(procedual1*albedo.r,lightMin,lightMax);
  }

  /*
 if (alpha0 ==255) {
                   alpha0 = int(floor(map(procedual1,0,255,roughMin,roughMax-16)));
 }
*/
  float noise = luma4(rnd)*255;  
 
    if(alpha0 >=  sssMin && alpha0 <=  sssMax)   alpha0 = int(clamp(alpha0,sssMin,sssMax)); // SSS

    if(alpha0 >=  lightMin && alpha0 <= lightMax)   alpha0 = int(clamp(alpha0+0,lightMin,lightMax)); // Emissives

    if(alpha0 >= roughMin && alpha0 <= roughMax)   alpha0 = int(clamp(alpha0+noise,roughMin,roughMax)); // Roughness


    if(alpha0 >= metalMin && alpha0 <= metalMax)   alpha0 = int(clamp(alpha0+noise,metalMin,metalMax)); // Metals

  noise /= 255;  


  float alpha1 = 0.0;
  float alpha2 = 0.0;

  if(alpha0 <= 128) alpha1 = floor(map( alpha0,  0, 128, 0, 255))/255;
  if(alpha0 >= 128) alpha2 = floor(map( alpha0,  128, 255, 0, 255))/255;

  //  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);

  float alpha3 = alpha1;
  float lm = lmx+(luma4(rnd*clamp(lmx*100,0,1)));
  if (res == 0.0f)    {
    lm = lmy+(luma4(rnd*clamp(lmy*100,0,1)));
    alpha3 = alpha2;
    //color.rgb = normal.rgb;
  }
  fragColor = color;
//  fragColor.rgb = test.rgb;
//if(int(textureLod(Sampler0, texCoord0,0).a*255)==255)alpha3 = 1.0;   
  fragColor.a = packUnorm2x4( alpha3,clamp(lm,0,0.95));
  
  
  }
