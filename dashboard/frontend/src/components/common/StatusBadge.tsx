/**
 * Status Badge Component
 */

import { Badge, BadgeProps } from '@mantine/core';
import {
  IconClock,
  IconPlayerPlay,
  IconCheck,
  IconX,
  IconMinus,
} from '@tabler/icons-react';

export type StatusType = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | string;

interface StatusBadgeProps extends Omit<BadgeProps, 'leftSection'> {
  status: StatusType;
  showIcon?: boolean;
}

const statusConfig: Record<string, { color: string; icon: React.ComponentType<{ size: number }> }> = {
  pending: { color: 'gray', icon: IconClock },
  running: { color: 'blue', icon: IconPlayerPlay },
  completed: { color: 'green', icon: IconCheck },
  failed: { color: 'red', icon: IconX },
  cancelled: { color: 'orange', icon: IconMinus },
};

export function StatusBadge({ status, showIcon = true, ...props }: StatusBadgeProps) {
  const config = statusConfig[status.toLowerCase()] || {
    color: 'gray',
    icon: IconMinus,
  };

  const Icon = config.icon;

  return (
    <Badge color={config.color} variant="light" {...props}>
      {showIcon && <Icon size={14} style={{ marginRight: 4 }} />}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </Badge>
  );
}

export default StatusBadge;
