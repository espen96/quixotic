#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;
uniform sampler2D Sampler1;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;

in mat4 ProjMat2;
in float vertexDistance;
in vec4 vertexColor;
in vec4 lightMapColor;
in vec4 overlayColor;
in vec2 texCoord0;
in vec4 normal;
in float lm;
out vec4 fragColor;
in float lmx;
in float lmy;
in vec4 glpos;

void main() {
  //vec3 rnd = ScreenSpaceDither(gl_FragCoord.xy);
  discardControlGLPos(gl_FragCoord.xy, glpos);
  vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;

  if(color.a * 255 <= 17.0) {
    discard;
  }
  color.rgb = mix(overlayColor.rgb, color.rgb, overlayColor.a);

  //  color *= lightMapColor;

  //color.rgb += rnd / 255;
  float mod2 = gl_FragCoord.x + gl_FragCoord.y;
  float res = mod(mod2, 2.0f);
  color.rgb = clamp(color.rgb, 0.0, 1);


if( vertexDistance < 1.5 && FogStart*0.000001 > 1) color.rgb = color.rgb;
  else if(res == 0.0f) {

    color.b =  clamp(lmx, 0, 0.95);
    color.r =  clamp(lmy, 0, 0.95);
  }

    #define sssMin 22
    #define sssMax 47
    #define lightMin 48
    #define lightMax 72
    #define roughMin 73
    #define roughMax 157
    #define metalMin 158
    #define metalMax 251

//  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);
  fragColor = color;

}
