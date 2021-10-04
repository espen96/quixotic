#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;

in float vertexDistance;
in vec2 texCoord0;
in vec4 vertexColor;
out vec4 fragColor;
in vec4 glpos;
void main() {
    discardControlGLPos(gl_FragCoord.xy, glpos);
    vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;
    vec3 rnd = ScreenSpaceDither( gl_FragCoord.xy );
    color.rgb +=rnd/255; 
  if (color.a*255 <= 17.0) {
    discard;
  }
    color.rgb = clamp(color.rgb,0.01,1);

//  fragColor = linear_fog(color, vertexDistance, FogStart, FogEnd, FogColor);
    fragColor = vec4(color.rgb,floor(color.a));
}
