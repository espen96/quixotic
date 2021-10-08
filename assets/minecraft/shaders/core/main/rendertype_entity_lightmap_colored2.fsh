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
    if (color.a < 0.1) {
        discard;
    }
//  fragColor = linear_fog(color, fragmentDistance, FogStart, FogEnd, FogColor);
    fragColor = color;
    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);
    color.rgb = clamp(color.rgb,0.01,1);



float lm = lmx;
  if (res == 0.0f)    {
    lm = lmy;
  }

   
//    fragColor.a = packUnorm2x4( 0,clamp(lm+(Bayer256(gl_FragCoord.xy)/16),0,0.9));


}
