#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform float OutSize;

in vec2 texCoord;

out vec4 fragColor;

void main() {
    // TODO: Add gametime
    int discardModulo = 1;
    int modulo = int(mod(gl_FragCoord.x + gl_FragCoord.y, 2.0));

    if (modulo == discardModulo) {
        ivec2 fragCoord = ivec2(gl_FragCoord.xy);

        float centerDepth = texelFetch(DiffuseDepthSampler, fragCoord, 0).r;
        if (centerDepth > 0.0001) {
            gl_FragDepth = centerDepth;
            fragColor = texelFetch(DiffuseSampler, fragCoord, 0);
            return;
        }

        vec4 depths = vec4(
            texelFetch(DiffuseDepthSampler, fragCoord + ivec2(0, 1), 0).r,
            texelFetch(DiffuseDepthSampler, fragCoord + ivec2(0, -1), 0).r,
            texelFetch(DiffuseDepthSampler, fragCoord + ivec2(1, 0), 0).r,
            texelFetch(DiffuseDepthSampler, fragCoord + ivec2(-1, 0), 0).r
        );
        float avgDepth = dot(depths, vec4(1)) / 4.0;
        vec4 weights = 1.0 - pow(depths - avgDepth, vec4(2.0));
        weights /= dot(weights, vec4(1));


        gl_FragDepth = dot(weights, depths);

        vec4 top    = texelFetch(DiffuseSampler, fragCoord + ivec2(0, 1), 0);
        vec4 bottom = texelFetch(DiffuseSampler, fragCoord + ivec2(0, -1), 0);
        vec4 right  = texelFetch(DiffuseSampler, fragCoord + ivec2(1, 0), 0);
        vec4 left   = texelFetch(DiffuseSampler, fragCoord + ivec2(-1, 0), 0);
        fragColor = weights.x * top + weights.y * bottom + weights.z * right + weights.w * left;
    } else {
        fragColor = texture(DiffuseSampler, texCoord);
        gl_FragDepth = texture(DiffuseDepthSampler, texCoord).r;
    }
}