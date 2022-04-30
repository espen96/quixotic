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
  discardControlGLPos(gl_FragCoord.xy, glpos);
  vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;
bool gui = isGUI( ProjMat2);
  if(color.a * 255 <= 17.0) {
    discard;
  }
  color.rgb = mix(overlayColor.rgb, color.rgb, overlayColor.a);


  float mod2 = gl_FragCoord.x + gl_FragCoord.y;
  float res = mod(mod2, 2.0f);
  color.rgb = clamp(color.rgb, 0.01, 1);


if( vertexDistance < 1.5 && FogStart*0.000001 > 1) color.rgb *= lightMapColor.rgb;
  else if(res == 0.0f && !gui) {
    color.b =  clamp(lmx, 0, 0.95);
    color.r =  clamp(lmy, 0, 0.95);
  }

  fragColor = color;


}
