#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;

in float vertexDistance;
in vec4 vertexColor;
in vec2 texCoord0;
in float lmx;
in float lmy;
out vec4 fragColor;

in vec4 glpos;
void main() {
    discardControlGLPos(gl_FragCoord.xy, glpos);
    vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;
    color.a = 1;
        float mod2 = gl_FragCoord.x + gl_FragCoord.y;
  float res = mod(mod2, 2.0f);
if( FogStart*0.000001 > 1) color.rgb = color.rgb;
  else if(res == 0.0f ) {
    color.b =  clamp(lmx, 0, 0.95);
    color.r =  clamp(lmy, 0, 0.95);
  }

    color.rgb = clamp(color.rgb, 0.01, 1);
//  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);
    fragColor = color;
}
