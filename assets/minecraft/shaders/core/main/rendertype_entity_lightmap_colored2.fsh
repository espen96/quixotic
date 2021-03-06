#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;
uniform sampler2D Sampler1;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;

in float vertexDistance;
in vec4 vertexColor;
in vec2 texCoord0;
in vec2 texCoord1;
in vec4 normal;
in vec4 glpos;
in mat4 ProjMat2;
out vec4 fragColor;
in float lmx;
in float lmy;

void main() {
  discardControlGLPos(gl_FragCoord.xy, glpos);
  vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;
  //  vec4 color = texture(Sampler0, texCoord0) * 1 * ColorModulator;
bool gui = isGUI( ProjMat2);
  if(color.a * 255 <= 17.0) {
    discard;
  }
//  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);

  float mod2 = gl_FragCoord.x + gl_FragCoord.y;
  float res = mod(mod2, 2.0f);

  float lm = lmx;
if( vertexDistance < 1.5 && FogStart*0.000001 > 1) color.rgb = color.rgb;
  else if(res == 0.0f && !gui) {
    lm = lmy;
    color.b =  clamp(lmx, 0, 0.95);
    color.r =  clamp(lmy, 0, 0.95);
  }
  fragColor = color;
  fragColor.a = 1;

}
