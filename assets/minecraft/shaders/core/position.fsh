#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;
uniform vec2 ScreenSize;
uniform mat4 ModelViewMat;
uniform mat4 ProjMat;

in mat4 gbufferModelViewInverse;
in float isSky;
in float vertexDistance;

out vec4 fragColor;

// at this point, the entire sky is drawable: isSky for sky, stars and void plane for everything else.
// similar logic can be added in vsh to separate void plane from stars.


vec3 renderSky(vec3 reddishTint, vec3 horizonColor, vec3 zenithColor, float h){

    h = 1.0-abs(h);

    float hsq = h*h;

    // gradient 1 = h^8
    float gradient1 = hsq*hsq;
    gradient1 *= gradient1;

    float gradient2 = 0.5*(hsq + h*hsq);
    
    horizonColor = mix(horizonColor, reddishTint, gradient1);
    return mix(zenithColor,horizonColor, gradient2 );
}


void main() {
    int index = inControl(gl_FragCoord.xy, ScreenSize.x);
    if (index != -1) {
        if (isSky > 0.5) {
            if (index >= 5 && index <= 15) {
                int c = (index - 5) / 4;
                int r = (index - 5) - c * 4;
                c = (c == 0 && r == 1) ? c : c + 1;
                fragColor = vec4(encodeFloat(ProjMat[c][r]), 1.0);
            } else if (index >= 16 && index <= 24) {
                int c = (index - 16) / 3;
                int r = (index - 16) - c * 3;
                fragColor = vec4(encodeFloat(ModelViewMat[c][r]), 1.0);
            } else if (index >= 3 && index <= 4) {
                fragColor = vec4(encodeFloat(atan(ProjMat[index - 3][index - 3])), 1.0);
            }  
            // store FogColor in control pixels
            else if (index == 25) {
                fragColor = vec4(FogColor.rgb,clamp(abs(FogStart)*0.01,0,1));
            
            }  
            // store SkyColor? in control pixels
            else if (index == 26) {
                fragColor = vec4(ColorModulator.rgb,1);
            }
            else if (index == 27) {
                fragColor = vec4(vec3(0.5),1);
            }

            else if (index == 28) {
                fragColor = vec4(1.0);
            }                             
            // blackout control pixels for sunDir so sun can write to them (by default, all pixels are FogColor)
            else {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }
        } else {
            discard;
        }
    }

    // not a control pixel, draw sky like normal
    else if (isSky > 0.5) {
        discard;

        /*
        vec4 screenPos = gl_FragCoord;
        screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
        screenPos.zw = vec2(1.0);
        vec3 view = normalize((gbufferModelViewInverse * screenPos).xyz);
        float ndusq = clamp(dot(view, vec3(0.0, 1.0, 0.0)), 0.0, 1.0);
        ndusq = ndusq * ndusq;
        fragColor = linear_fog(ColorModulator, pow(1.0 - ndusq, 8.0), 0.0, 1.0, FogColor);

//   fragColor.rgb = renderSky(ColorModulator.rgb,ColorModulator.rgb,FogColor.rgb,view.y);

*/
    }/*
    else {
        fragColor = linear_fog(ColorModulator, vertexDistance, FogStart, FogEnd, FogColor);

    }
*/
}
